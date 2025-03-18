import requests
import imageio
import os
import numpy as np
import cv2
from datetime import datetime, timedelta
import tempfile
import zipfile
import fiona
import fiona.crs
import simplekml
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as ReportLabImage, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

app = Flask(__name__)
CORS(app)  # Allow all origins during development

# Sentinel API Credentials
CLIENT_ID = "d05a7096-ebca-48da-8277-3dce9f1e92ee"
CLIENT_SECRET = "VoSk8FubfKwH9aqarig7WgEeExBJzRGO"

# Create static folder if it doesn't exist
os.makedirs("static", exist_ok=True)

# Function to Get Sentinel API Token
def get_sentinel_token():
    token_url = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
    data = {"grant_type": "client_credentials", "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}

    try:
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        token = response.json().get("access_token")
        print("✅ Token fetched successfully")
        return token
    except requests.exceptions.RequestException as e:
        print(f"❌ Token fetch error: {e}")
        return None

# Function to Check if an Image is Blank (Black)
def is_black_image(image_path):
    image = imageio.imread(image_path)
    return np.all(image == 0)  # Returns True if all pixels are black

# Function to Apply Brightness, Exposure, Contrast, Highlights Adjustments
def apply_image_adjustments(image_path):
    img = Image.open(image_path)

    # Increase Brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.5)  # Equivalent to +100 brightness

    # Increase Exposure (simulating by increasing brightness again)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.8)  # Equivalent to +100 exposure

    # Adjust Contrast (-12)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(0.88)  # Scaling down contrast slightly

    # Adjust Highlights (Boosting Whites)
    img_cv = np.array(img)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.add(l, 50)  # Increasing L-channel for highlights
    lab = cv2.merge((l, a, b))
    img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Convert back to PIL Image and Save
    img = Image.fromarray(img_cv)
    img.save(image_path)

# Function to detect changes between two satellite images
def detect_changes(before_image_path, after_image_path):
    """
    Detects changes between two satellite images and returns a change mask
    highlighting vegetation, urban development, and water bodies.
    """
    # Read images
    before_img = cv2.imread(before_image_path)
    after_img = cv2.imread(after_image_path)

    # Convert to same size if different
    if before_img.shape != after_img.shape:
        after_img = cv2.resize(after_img, (before_img.shape[1], before_img.shape[0]))

    # Convert to HSV color space for better color analysis
    before_hsv = cv2.cvtColor(before_img, cv2.COLOR_BGR2HSV)
    after_hsv = cv2.cvtColor(after_img, cv2.COLOR_BGR2HSV)

    # Create masks for each category
    # Vegetation: Focus on green colors
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    before_veg_mask = cv2.inRange(before_hsv, lower_green, upper_green)
    after_veg_mask = cv2.inRange(after_hsv, lower_green, upper_green)

    # Urban/Built-up: Focus on grayscale values (buildings, roads)
    before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
    before_urban_mask = cv2.adaptiveThreshold(before_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
    after_urban_mask = cv2.adaptiveThreshold(after_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)

    # Water bodies: Focus on blue colors
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    before_water_mask = cv2.inRange(before_hsv, lower_blue, upper_blue)
    after_water_mask = cv2.inRange(after_hsv, lower_blue, upper_blue)

    # Detect changes
    veg_change = cv2.bitwise_xor(before_veg_mask, after_veg_mask)
    urban_change = cv2.bitwise_xor(before_urban_mask, after_urban_mask)
    water_change = cv2.bitwise_xor(before_water_mask, after_water_mask)

    # Create a composite change mask
    change_mask = np.zeros((before_img.shape[0], before_img.shape[1], 3), dtype=np.uint8)

    # Green for vegetation changes
    change_mask[veg_change > 0] = [0, 255, 0]

    # Red for urban changes
    change_mask[urban_change > 0] = [255, 0, 0]

    # Blue for water changes
    change_mask[water_change > 0] = [0, 0, 255]

    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)
    change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)

    return change_mask

# Function to create an overlay image with changes highlighted
def create_overlay_image(image_path, change_mask, output_path, alpha=0.5):
    """
    Creates an overlay image with changes highlighted on the original image.
    """
    # Read original image
    original = cv2.imread(image_path)

    # Create a copy of the original image
    overlay = original.copy()

    # Apply the change mask to the overlay
    overlay = cv2.addWeighted(overlay, 1 - alpha, change_mask, alpha, 0)

    # Add a legend to the image
    # Convert from BGR to RGB for PIL
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    overlay_pil = Image.fromarray(overlay_rgb)
    draw = ImageDraw.Draw(overlay_pil)

    # Add legend
    legend_height = 20
    legend_width = 150
    legend_margin = 10

    # Draw legend box
    draw.rectangle(
        [(legend_margin, legend_margin),
         (legend_margin + legend_width, legend_margin + 3 * legend_height)],
        fill=(255, 255, 255, 180),
        outline=(0, 0, 0)
    )

    # Add legend items
    try:
        # Try to use a system font
        font = ImageFont.truetype("Arial", 12)
    except IOError:
        # Fallback to default font
        font = ImageFont.load_default()

    # Legend items
    draw.rectangle(
        [(legend_margin + 5, legend_margin + 5),
         (legend_margin + 15, legend_margin + 15)],
        fill=(0, 255, 0)
    )
    draw.text((legend_margin + 20, legend_margin + 2), "Vegetation", font=font, fill=(0, 0, 0))

    draw.rectangle(
        [(legend_margin + 5, legend_margin + legend_height + 5),
         (legend_margin + 15, legend_margin + legend_height + 15)],
        fill=(255, 0, 0)
    )
    draw.text((legend_margin + 20, legend_margin + legend_height + 2), "Urban", font=font, fill=(0, 0, 0))

    draw.rectangle(
        [(legend_margin + 5, legend_margin + 2 * legend_height + 5),
         (legend_margin + 15, legend_margin + 2 * legend_height + 15)],
        fill=(0, 0, 255)
    )
    draw.text((legend_margin + 20, legend_margin + 2 * legend_height + 2), "Water", font=font, fill=(0, 0, 0))

    # Convert back to OpenCV format and save
    overlay = cv2.cvtColor(np.array(overlay_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, overlay)

    return output_path

# Function to generate advanced visualization using matplotlib
def generate_change_visualization(before_path, after_path, output_path):
    """
    Generates a more advanced visualization showing before/after with changes highlighted.
    """
    # Read images
    before_img = cv2.imread(before_path)
    after_img = cv2.imread(after_path)
    before_rgb = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
    after_rgb = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)

    # Detect changes
    change_mask = detect_changes(before_path, after_path)
    change_mask_rgb = cv2.cvtColor(change_mask, cv2.COLOR_BGR2RGB)

    # Create a figure with three subplots
    plt.figure(figsize=(15, 5))

    # Plot before image
    plt.subplot(1, 3, 1)
    plt.imshow(before_rgb)
    plt.title('Before')
    plt.axis('off')

    # Plot after image
    plt.subplot(1, 3, 2)
    plt.imshow(after_rgb)
    plt.title('After')
    plt.axis('off')

    # Plot change mask
    plt.subplot(1, 3, 3)

    # Create a blended image
    alpha = 0.7
    blended = cv2.addWeighted(after_rgb, 1 - alpha, change_mask_rgb, alpha, 0)
    plt.imshow(blended)
    plt.title('Change Analysis')
    plt.axis('off')

    # Add a legend
    legend_elements = [
        mpatches.Patch(color='green', label='Vegetation'),
        mpatches.Patch(color='red', label='Urban'),
        mpatches.Patch(color='blue', label='Water')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path

# Fetch Satellite Images with Enhanced Processing
def fetch_satellite_images(polygon_coords, start_date, end_date):
    token = get_sentinel_token()
    if not token:
        print("❌ Token not available!")
        return []

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    images = []
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"🔍 Trying to fetch image for: {date_str}")

        payload = {
            "input": {
                "bounds": {"geometry": {"type": "Polygon", "coordinates": polygon_coords}},
                "data": [
                    {
                        "type": "sentinel-2-l2a",  # Higher quality images
                        "dataFilter": {
                            "timeRange": {"from": f"{date_str}T00:00:00Z", "to": f"{date_str}T23:59:59Z"},
                            "maxCloudCoverage": 10  # Maximum cloud cover allowed
                        }
                    }
                ]
            },
            "output": {"width": 512, "height": 512, "format": "png"},
            "evalscript": """
                function setup() {
                    return {
                        input: ["B04", "B03", "B02"],
                        output: { bands: 3, sampleType: "UINT8" }
                    };
                }
                function evaluatePixel(sample) {
                    return [
                        sample.B04 * 255,
                        sample.B03 * 255,
                        sample.B02 * 255
                    ];
                }
            """
        }

        try:
            response = requests.post("https://services.sentinel-hub.com/api/v1/process", headers=headers, json=payload)
            response.raise_for_status()

            img_filename = f"{date_str}.png"
            img_path = os.path.join("static", img_filename)

            with open(img_path, "wb") as img_file:
                img_file.write(response.content)

            # Check if the image is black
            if is_black_image(img_path):
                print(f"⚠️ Black image detected for {date_str}, skipping...")
                os.remove(img_path)  # Delete black images
                current_date += timedelta(days=1)  # Move to the next day
                continue

            # Apply Adjustments
            apply_image_adjustments(img_path)

            images.append(img_filename)
            print(f"✅ Image saved: {img_path}")

            current_date += timedelta(days=3)  # Increase frequency for more images
        except requests.exceptions.RequestException as e:
            print(f"❌ API Error for {date_str}: {e}")
            current_date += timedelta(days=1)  # Try the next day instead of skipping

    return images

# Create Timelapse GIF
def create_gif(image_filenames, output_path="static/timelapse.gif"):
    if not image_filenames:
        print("❌ No images to create a GIF!")
        return None

    image_paths = [os.path.join("static", img) for img in image_filenames]

    # Ensure all images exist
    for path in image_paths:
        if not os.path.exists(path):
            print(f"❌ Missing image: {path}")
            return None

    try:
        images = [imageio.imread(img) for img in image_paths]
        # Set loop=0 to make the GIF loop indefinitely
        imageio.mimsave(output_path, images, duration=0.5, loop=0)
        print(f"🎥 GIF created: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error creating GIF: {e}")
        return None

# Create Timelapse Video
def create_video(image_filenames, output_path="static/timelapse.mp4", fps=2):
    if not image_filenames:
        print("❌ No images to create a video!")
        return None

    image_paths = [os.path.join("static", img) for img in image_filenames]

    # Ensure all images exist
    for path in image_paths:
        if not os.path.exists(path):
            print(f"❌ Missing image: {path}")
            return None

    try:
        # Get dimensions from first image
        first_img = cv2.imread(image_paths[0])
        height, width, layers = first_img.shape
        size = (width, height)

        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
        out = cv2.VideoWriter(output_path, fourcc, fps, size)

        # Add images to video
        for img_path in image_paths:
            img = cv2.imread(img_path)
            out.write(img)

            # Add transition frame (hold each frame for a bit)
            for _ in range(3):  # Add 3 duplicate frames for a pause effect
                out.write(img)

        out.release()
        print(f"🎥 Video created: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error creating video: {e}")
        return None

# Modified function to generate PDF report with change analysis
def generate_pdf_report(image_filenames, polygon_coords, start_date, end_date):
    """
    Generate a PDF report showing satellite imagery changes with analysis overlays
    """
    if not image_filenames or len(image_filenames) < 4:
        print("❌ Not enough images to generate a report")
        return None

    # Sort images by date
    image_filenames.sort()

    # Select the first and last images for change analysis
    first_image = image_filenames[0]
    last_image = image_filenames[-1]

    # Select images for display
    first_image_path = os.path.join("static", first_image)
    last_image_path = os.path.join("static", last_image)

    # Generate change analysis visualization
    analysis_path = os.path.join("static", "change_analysis.png")
    generate_change_visualization(first_image_path, last_image_path, analysis_path)

    # Create overlays for each pair of images
    middle_idx = len(image_filenames) // 2
    second_image = image_filenames[middle_idx // 2]
    third_image = image_filenames[middle_idx]
    fourth_image = image_filenames[-1]

    # Generate overlays
    overlay1_path = os.path.join("static", "overlay1.png")
    overlay2_path = os.path.join("static", "overlay2.png")

    # Create overlays between first and middle, and middle and last images
    first_middle_mask = detect_changes(
        os.path.join("static", first_image),
        os.path.join("static", third_image)
    )
    middle_last_mask = detect_changes(
        os.path.join("static", third_image),
        os.path.join("static", fourth_image)
    )

    create_overlay_image(
        os.path.join("static", third_image),
        first_middle_mask,
        overlay1_path
    )
    create_overlay_image(
        os.path.join("static", fourth_image),
        middle_last_mask,
        overlay2_path
    )

    # Set up the PDF document - use landscape orientation
    report_path = "static/satellite_report.pdf"
    doc = SimpleDocTemplate(
        report_path,
        pagesize=landscape(letter),
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    # Create styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    title_style.alignment = 1  # Center alignment
    subtitle_style = styles['Heading2']
    subtitle_style.alignment = 1
    normal_style = styles['Normal']

    # Create a style for Arabic text if needed
    arabic_style = ParagraphStyle(
        'ArabicStyle',
        parent=styles['Normal'],
        alignment=2,  # Right alignment for Arabic
        fontName='Helvetica-Bold',
        fontSize=12
    )

    # Create a footer style
    footer_style = ParagraphStyle(
        'FooterStyle',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.gray
    )

    # Create PDF content
    content = []

    # Add title
    content.append(Paragraph("Satellite Imagery Change Analysis Report", title_style))
    content.append(Spacer(1, 0.25*inch))

    # Add date information and coordinates in a table format
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create header information table
    header_data = [
        ["Report Generated:", report_date, "Analysis Period:", f"{start_date} to {end_date}"],
        ["Location:", f"Coordinates: {polygon_coords[0][0]}", "Analysis Type:", "Vegetation, Urban, Water Changes"]
    ]

    header_table = Table(header_data, colWidths=[1.5*inch, 2.5*inch, 1.5*inch, 2.5*inch])
    header_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('BACKGROUND', (2, 0), (2, -1), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
    ]))

    content.append(header_table)
    content.append(Spacer(1, 0.5*inch))

    # Add the analysis visualization
    content.append(Paragraph("Satellite Image Change Analysis Overview", subtitle_style))
    content.append(Spacer(1, 0.25*inch))

    # Add the analysis image
    if os.path.exists(analysis_path):
        analysis_img = ReportLabImage(analysis_path, width=8*inch, height=2.5*inch)
        content.append(analysis_img)
    else:
        content.append(Paragraph("Analysis visualization could not be generated", normal_style))

    content.append(Spacer(1, 0.5*inch))

    # Add detailed change analysis
    content.append(Paragraph("Detailed Change Analysis", subtitle_style))
    content.append(Spacer(1, 0.25*inch))

    # Create a table for the detailed images
    image_table_data = [
        [Paragraph(f"Base Image ({first_image.replace('.png', '')})", normal_style),
         Paragraph(f"Mid-Period Changes ({third_image.replace('.png', '')})", normal_style)],
        [ReportLabImage(os.path.join("static", first_image), width=4*inch, height=3*inch),
         ReportLabImage(overlay1_path, width=4*inch, height=3*inch)],
        [Paragraph(f"Mid-Period Image ({third_image.replace('.png', '')})", normal_style),
         Paragraph(f"Latest Period Changes ({fourth_image.replace('.png', '')})", normal_style)],
        [ReportLabImage(os.path.join("static", third_image), width=4*inch, height=3*inch),
         ReportLabImage(overlay2_path, width=4*inch, height=3*inch)]
    ]

    image_table = Table(image_table_data, colWidths=[4.5*inch, 4.5*inch])
    image_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('BACKGROUND', (0, 2), (-1, 2), colors.lightgrey),
    ]))

    content.append(image_table)
    content.append(Spacer(1, 0.5*inch))

    # Add analysis legend
    content.append(Paragraph("Change Analysis Legend", subtitle_style))

    legend_data = [
        ["Color", "Change Type", "Description"],
        ["Green", "Vegetation Changes", "Areas where vegetation has increased or decreased"],
        ["Red", "Urban Development", "New construction, buildings, or infrastructure"],
        ["Blue", "Water Bodies", "Changes in water levels, reservoirs, or water features"]
    ]

    legend_table = Table(legend_data, colWidths=[1*inch, 2*inch, 6*inch])
    legend_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 1), (0, 1), colors.green),
        ('BACKGROUND', (0, 2), (0, 2), colors.red),
        ('BACKGROUND', (0, 3), (0, 3), colors.blue),
    ]))

    content.append(legend_table)

    # Add a footer with disclaimer
    disclaimer_text = "Disclaimer: This is not an official map. The satellite imagery and analysis are for informational purposes only."
    footer_text = f"{disclaimer_text} | Generated using Sentinel-2 imagery | Analysis period: {start_date} to {end_date}"

    def add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(inch, 0.5*inch, footer_text)
        canvas.restoreState()

    # Build the PDF with the footer function
    doc.build(content, onFirstPage=add_footer, onLaterPages=add_footer)
    print(f"✅ PDF report generated: {report_path}")
    return report_path

# Endpoint to Get Images
@app.route("/get_images", methods=["POST"])
def get_images():
    data = request.json
    polygon = data.get("polygon")
    start_date = data.get("start_date")
    end_date = data.get("end_date")

    if not polygon or not start_date or not end_date:
        return jsonify({"error": "Missing required parameters"}), 400

    image_filenames = fetch_satellite_images(polygon, start_date, end_date)

    # Check for the existence of additional files
    additional_files = {}
    static_folder = "static"
    additional_files["timelapse_gif"] = os.path.exists(os.path.join(static_folder, "timelapse.gif"))
    additional_files["timelapse_video"] = os.path.exists(os.path.join(static_folder, "timelapse.mp4"))
    additional_files["kml"] = os.path.exists(os.path.join(static_folder, "polygon.kml"))

    if image_filenames:
        # Generate URLs for each image
        image_urls = [f"/static/{img}" for img in image_filenames]
        return jsonify({"images": image_filenames, "image_urls": image_urls, "additional_files": additional_files})
    else:
        return jsonify({"error": "No clear images found", "additional_files": additional_files}), 404

# Endpoint to Generate Timelapse GIF
@app.route("/generate_timelapse", methods=["POST"])
def generate_timelapse():
    data = request.json
    polygon = data.get("polygon")
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    existing_images = data.get("existing_images")

    if not polygon or not start_date or not end_date:
        return jsonify({"error": "Missing required parameters"}), 400

    # Use existing images if provided, otherwise fetch new ones
    image_filenames = existing_images or fetch_satellite_images(polygon, start_date, end_date)

    if not image_filenames:
        return jsonify({"error": "No clear images found for timelapse"}), 404

    # Create GIF from the images
    gif_path = create_gif(image_filenames)

    if gif_path and os.path.exists(gif_path):
        return jsonify({"success": True, "gif_url": "/static/timelapse.gif"})
    else:
        return jsonify({"error": "Failed to create timelapse"}), 500

# Endpoint to Generate Timelapse Video
@app.route("/generate_video", methods=["POST"])
def generate_video():
    data = request.json
    polygon = data.get("polygon")
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    existing_images = data.get("existing_images")
    fps = data.get("fps", 2)  # Default to 2 FPS if not provided

    if not polygon or not start_date or not end_date:
        return jsonify({"error": "Missing required parameters"}), 400

    # Use existing images if provided, otherwise fetch new ones
    image_filenames = existing_images or fetch_satellite_images(polygon, start_date, end_date)

    if not image_filenames:
        return jsonify({"error": "No clear images found for video"}), 404

    # Create video from the images
    video_path = create_video(image_filenames, fps=fps)

    if video_path and os.path.exists(video_path):
        return jsonify({"success": True, "video_url": "/static/timelapse.mp4"})
    else:
        return jsonify({"error": "Failed to create video"}), 500

# Endpoint to download media file (GIF or MP4)
@app.route("/download_media", methods=["GET"])
def download_media():
    media_type = request.args.get("type", "gif")  # Default to GIF if not specified

    if media_type == "gif":
        file_path = "static/timelapse.gif"
        mimetype = "image/gif"
        filename = "satellite_timelapse.gif"
    elif media_type == "video":
        file_path = "static/timelapse.mp4"
        mimetype = "video/mp4"
        filename = "satellite_timelapse.mp4"
    else:
        return jsonify({"error": "Invalid media type"}), 400

    if not os.path.exists(file_path):
        return jsonify({"error": f"No {media_type} file found"}), 404

    return send_file(file_path, mimetype=mimetype, as_attachment=True, download_name=filename)

# Endpoint to download shapefile
@app.route("/download_shapefile", methods=["POST"])
def download_shapefile():
    data = request.json
    polygon_coords = data.get("polygon")

    if not polygon_coords:
        return jsonify({"error": "Missing polygon coordinates"}), 400

    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Define shapefile path
            shp_path = os.path.join(tmpdir, "polygon.shp")

            # Create the shapefile
            schema = {
                'geometry': 'Polygon',
                'properties': {'id': 'int'},
            }

            with fiona.open(shp_path, 'w',
                          driver='ESRI Shapefile',
                          crs=fiona.crs.from_epsg(4326),
                          schema=schema) as c:
                # Create a feature
                c.write({
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': polygon_coords
                    },
                    'properties': {'id': 1},
                })

            # Create a zip file containing all shapefile components
            zip_path = os.path.join(tmpdir, "polygon_shapefile.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # Add all related files
                for ext in ['.shp', '.shx', '.dbf', '.prj']:
                    file_path = os.path.join(tmpdir, f"polygon{ext}")
                    if os.path.exists(file_path):
                        zipf.write(file_path, os.path.basename(file_path))

            return send_file(zip_path, as_attachment=True, download_name="polygon_shapefile.zip")
    except Exception as e:
        print(f"Error creating shapefile: {e}")
        return jsonify({"error": "Failed to create shapefile"}), 500

# Endpoint to download KML
@app.route("/download_kml", methods=["POST"])
def download_kml():
    data = request.json
    polygon_coords = data.get("polygon")

    if not polygon_coords:
        return jsonify({"error": "Missing polygon coordinates"}), 400

    try:
        # Create KML
        kml = simplekml.Kml()
        pol = kml.newpolygon(name="Selected Area")

        # Set polygon coordinates
        # KML requires longitude, latitude order
        kml_coords = []
        for coord in polygon_coords[0]:  # Use the first ring (outer boundary)
            # Swap lon/lat for KML if needed - depends on how your frontend provides coordinates
            kml_coords.append((coord[0], coord[1]))

        pol.outerboundaryis = kml_coords

        # Create a temporary file for the KML
        kml_path = os.path.join("static", "polygon.kml")
        kml.save(kml_path)

        return send_file(kml_path, as_attachment=True, download_name="polygon.kml")
    except Exception as e:
        print(f"Error creating KML: {e}")
        return jsonify({"error": "Failed to create KML"}), 500

# Endpoint to list existing images
@app.route("/list_existing_images", methods=["GET"])
def list_existing_images():
    try:
        # List all files in the static directory
        files = os.listdir("static")

        # Filter for image files (.png)
        image_files = [f for f in files if f.endswith('.png')]

        # Sort files by date (assuming filename format is YYYY-MM-DD.png)
        image_files.sort()

        # Check for additional files
        additional_files = {
            "timelapse_gif": "timelapse.gif" in files,
            "timelapse_video": "timelapse.mp4" in files,
            "kml": "polygon.kml" in files
        }

        return jsonify({
            "images": image_files,
            "additional_files": additional_files
        })

    except Exception as e:
        print(f"Error listing existing images: {e}")
        return jsonify({"error": "Failed to list existing images"}), 500

# Modify the generate_report endpoint to use the enhanced function
@app.route("/generate_report", methods=["POST"])
def generate_report():
    data = request.json
    polygon = data.get("polygon")
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    existing_images = data.get("existing_images")

    if not polygon or not start_date or not end_date:
        return jsonify({"error": "Missing required parameters"}), 400

    # Use existing images if provided, otherwise fetch new ones
    image_filenames = existing_images or fetch_satellite_images(polygon, start_date, end_date)

    if not image_filenames or len(image_filenames) < 4:
        return jsonify({"error": "Not enough images found for change analysis"}), 404

    # Generate the enhanced PDF report
    report_path = generate_pdf_report(image_filenames, polygon, start_date, end_date)

    if report_path and os.path.exists(report_path):
        return jsonify({
            "success": True,
            "report_url": "/static/satellite_report.pdf"
        })
    else:
        return jsonify({"error": "Failed to generate report"}), 500

# Endpoint to download the PDF report
@app.route("/download_report", methods=["GET"])
def download_report():
    report_path = "static/satellite_report.pdf"

    if not os.path.exists(report_path):
        return jsonify({"error": "Report not found"}), 404

    return send_file(
        report_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="satellite_change_report.pdf"
    )

if __name__ == "__main__":
    app.run(debug=True)
