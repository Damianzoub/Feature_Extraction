from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import folium
import os
import pandas as pd
from collections import defaultdict
import csv
import sys
import numpy as np
from functionalities import *
import plotly.graph_objects as go
import plotly.io as pio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformation import DataTransformer


app = FastAPI()

# Mount static and template folders
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Read ais data
data = pd.read_csv(r"C:\Users\user\Documents\Feature_Extraction\Feature_Extraction\ais.csv")
trajectories = list(data.shipid.unique())

trajectory_data = defaultdict(list)

# Open csv to display trajectories
with open(r"C:\Users\user\Documents\Feature_Extraction\Feature_Extraction\ais.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tid = row["shipid"]
        lat = float(row["lat"])
        lon = float(row["lon"])
        trajectory_data[tid].append((lat, lon))


# Display ship selector
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Send list of trajectory IDs to template
    trajectory_ids = list(trajectory_data.keys())
    return templates.TemplateResponse("index.html", {"request": request, "trajectory_ids": trajectory_ids})


# Main
@app.get("/trajectory/{tid}")
async def get_trajectory(tid: str):
    # Print error message if something goes wrong
    if tid not in trajectory_data:
        return JSONResponse({"error": "Trajectory not found"}, status_code=404)

    # Get trajectory of a specific ship
    trajectory = trajectory_data[tid]

    # Get the full information from ais.csv regarding the ship_id==tid and save it in order to extract mfs
    df = data.loc[data.shipid == tid]
    df.to_csv("trajectory.csv", index=False)

    # Extract mfs for a specific ship
    data_transform = DataTransformer(
        dataset_path='trajectory.csv',
        time_col='t',
        id_col='shipid',
        speed_col='speed',
        heading_col='heading',
        lat_col='lat',
        long_col='lon',
        course_col='course',
        shiptype_col='shiptype',
        destination_col='destination',
        numeric_columns=['heading', 'course', 'speed'],
        categorical_columns=['shiptype', 'destination']
    )

    data_transform.load_data()
    data_transform.transform_dataset()

    features_df = data_transform.get_all_features()

    # Get mfs to send them to index.html
    stats = {
        "num_points": len(trajectory),
        "start": (round(trajectory[0][0], 2), round(trajectory[0][1], 2)),
        "end": (round(trajectory[-1][0], 2), round(trajectory[-1][1], 2)),
        "max_speed": round(features_df.max_speed.values[0], 2),
        "min_speed": round(features_df.min_speed.values[0], 2),
        "avg_speed": round(features_df.avg_speed.values[0], 2),
        "std_speed": round(features_df.std_speed.values[0], 2),
        "max_acc": round(features_df.max_acceleration.values[0], 2),
        "min_acc": round(features_df.min_acceleration.values[0], 2),
        "avg_acc": round(features_df.avg_acceleration.values[0], 2),
        "std_acc": round(features_df.std_speed.values[0], 2),
        "mean_rot": round(features_df.rot_mean.values[0], 2),
        "std_rot": round(features_df.rot_std.values[0], 2)
    }

    # Put -1 to mfs that are inf or nan
    for x in stats.keys():
        if np.isnan(np.array(stats[x])).any() or np.isinf(np.array(stats[x])).any():
            stats[x] = -1

    # Compute bounds for map
    lats = [lat for lat, lon in trajectory]
    lons = [lon for lat, lon in trajectory]
    # Add an extra term
    sw = [min(lats)-0.005, min(lons)-0.005]
    ne = [max(lats)+0.005, max(lons)+0.005]

    # Create Folium map and fit bounds
    m = folium.Map(location=[(sw[0]+ne[0])/2, (sw[1]+ne[1])/2])
    m.fit_bounds([sw, ne])

    folium.Marker(trajectory[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(trajectory[-1], tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    folium.PolyLine(trajectory, color="blue", weight=4.5, opacity=0.8).add_to(m)

    # Save the map html with the trajectory id
    map_path = os.path.join("static", f"map_{tid}.html")
    m.save(map_path)

    # Return the mfs and the map to index.html
    return {
        "stats": stats,
        "map_url": f"/static/map_{tid}.html"
    }


@app.get("/trajectory/{tid}/plot", response_class=HTMLResponse)
async def plot_trajectory_timeseries(request: Request, tid: str = None):
    plot_html_file = None

    if tid not in trajectory_data:
        return HTMLResponse(content="Trajectory not found", status_code=404)

    df = speed(data.loc[data.shipid == tid])
    timestamps = pd.to_datetime(df["t"])
    speeds = df["Speed"]

    accelerations = np.gradient(speeds, edge_order=2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=speeds, mode='lines', name='Speed (km/h)'))
    fig.add_trace(go.Scatter(x=timestamps, y=accelerations, mode='lines', name='Acceleration (km/h^2)'))
    fig.update_layout(xaxis_title="Time", yaxis_title="Value")

    fig.update_layout(
        autosize=True,
        height=600,
    )
    fig.write_html("static/plot.html", include_plotlyjs='cdn', full_html=True)

    plot_filename = f"plot_{tid}.html"
    plot_path = os.path.join("static", plot_filename)
    pio.write_html(fig, file=plot_path, auto_open=False, include_plotlyjs='cdn')

    plot_html_file = f"/static/{plot_filename}"

    # Now render the template with the URL of the saved plot image
    return templates.TemplateResponse("index.html", {
        "request": request,
        "trajectory_id": tid,
        "plot_html_file": plot_html_file
    })


"""
To run it:
cd UI
uvicorn app:app --host 0.0.0.0 --port 8081 --reload
"""
