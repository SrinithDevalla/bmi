from flask import Flask, render_template
import plotly.express as px
import plotly.graph_objects as go
from data_processing import df, anomalies, traffic_by_hour
from traffic_api import distance_data
import pandas as pd

app = Flask(__name__)

@app.route("/")
def traffic_dashboard():
    df_html = df.head(50).to_html(classes="table table-striped")

    fig1 = px.line(traffic_by_hour, x=traffic_by_hour.index, y=traffic_by_hour.values,
                   labels={'x': 'Hour', 'y': 'Avg Density'}, title="🚦 Peak Traffic Hours")
    graph1 = fig1.to_html(full_html=False)

    fig2 = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color="Traffic_Cluster",
                             zoom=12, height=500, mapbox_style="carto-positron",
                             title="Traffic Clustering Map")
    graph2 = fig2.to_html(full_html=False)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df.index, y=df["Traffic_Density"], mode='lines', name="Density"))
    fig3.add_trace(go.Scatter(x=anomalies.index, y=anomalies["Traffic_Density"], mode='markers',
                              name="Anomalies", marker=dict(color='red', size=8)))
    fig3.update_layout(title=" Anomaly Detection in Traffic Data")
    graph3 = fig3.to_html(full_html=False)

    flyovers = pd.read_csv("flyover_suggestions.csv")
    flyovers_html = flyovers.to_html(classes="table table-bordered")
     # Load Road Widening Data
    widening_df = pd.read_csv("road_widening_suggestions.csv")
    widening_html = widening_df.to_html(classes="table table-hover")

    return render_template("dashboard.html", df_table=df_html, graph1=graph1,
                           graph2=graph2, graph3=graph3,
                           flyovers=flyovers_html, road_widening=widening_html)

if __name__ == "__main__":
    app.run(debug=True)
