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
from PIL import Image, ImageEnhance

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
        # Add this new endpoint to your Flask application (app.py)

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


if __name__ == "__main__":
    app.run(debug=True)
