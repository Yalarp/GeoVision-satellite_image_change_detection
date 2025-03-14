"# GeoVision-satellite_image_change_detection" 

# Satellite Data Viewer

The Satellite Data Viewer is a web application that allows users to view and analyze satellite imagery for a specified region and date range. The application uses Flask for the backend and Leaflet for the frontend map interface.

## Features

- View satellite images for a specified region and date range.
- Generate timelapse GIFs and videos from the fetched images.
- Download shapefiles and KML files of the selected region.
- Load and display existing images from the static folder.
- Enlarge images for better viewing.

## Installation

### Prerequisites

- Python 3.x
- Node.js (for running the frontend)

### Steps

1. **Clone the Repository**

   ```sh
   git clone https://github.com/your-username/satellite-data-viewer.git
   cd satellite-data-viewer
   ```

2. **Set Up the Backend**

   - Create a virtual environment and activate it:

     ```sh
     python -m venv venv
     source venv/bin/activate  # On Windows use `venv\Scripts\activate`
     ```

   - Install the required Python packages:

     ```sh
     pip install flask requests imageio numpy opencv-python-headless fiona simplekml flask-cors pillow
     ```

3. **Set Up the Frontend**

   - Ensure you have Node.js installed.
   - Install the required frontend dependencies (if any).

4. **Run the Application**

   - Start the Flask server:

     ```sh
     python app.py
     ```

   - Open your browser and navigate to `http://127.0.0.1:5000` to access the application.

## Usage

1. **Select a Region on the Map**

   - Use the drawing tools to select a region of interest on the map.

2. **Set the Date Range**

   - Enter the start and end dates for which you want to fetch satellite images.

3. **Fetch Images**

   - Click the "Get Images" button to fetch satellite images for the selected region and date range.
   - The images will be displayed in the gallery section.

4. **Generate Timelapse GIF or Video**

   - Click the "Generate Timelapse GIF" or "Generate Video" button to create a timelapse GIF or video from the fetched images.
   - The generated timelapse will be displayed in the respective tab.

5. **Download Shapefile or KML**

   - Click the "Download Shapefile" or "Download KML" button to download the selected region as a shapefile or KML file.

6. **Load Existing Images**

   - Click the "Load Existing Images" button to load and display existing images from the static folder.

7. **Enlarge Images**

   - Click on any image in the gallery to enlarge it for better viewing.

## File Structure

- `app.py`: The main Flask application file.
- `index.html`: The main HTML file for the frontend.
- `styles.css`: The CSS file for styling the frontend.
- `static/`: Directory to store static files like images.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Leaflet](https://leafletjs.com/) for the map interface.
- [Flask](https://flask.palletsprojects.com/) for the backend framework.
- [Sentinel Hub](https://www.sentinel-hub.com/) for satellite imagery.

## Contact

For any questions or support, please contact [2706pralay@gmail.com](mailto:2706pralay@gmail.com).
```
