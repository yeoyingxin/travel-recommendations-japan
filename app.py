import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

import os
import pandas as pd
from openpyxl import load_workbook
from datetime import datetime

from data_ingestion import qd_client
from retrieval import rag_hybrid_search 

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Container([
    html.H1("Travelling to Japan? Get some recommendations here!"),
    dbc.Input(id="query", type="text", placeholder="Enter your question", style={"width": "60%"}),
    html.Button("Search", id="search-btn"),
    html.Br(), html.Br(),
    html.Div(id="results")
    ])
])

@app.callback(
    Output("results", "children"), 
    Input("search-btn", "n_clicks"),
    State("query", "value"))
def answer_query(n_clicks, query):

    if not n_clicks:
        return ""

    # user clicked but didn’t type anything
    if not query or query.strip() == "":
        return "⚠️ Please type in your question."

    # valid input
    answer, retrieved_doc = rag_hybrid_search(query)
    response = f"Answer for: {query}\n\n Answer: {answer}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    # save response and document retrieved
    log_file = "data/query_log.csv"
    new_data = pd.DataFrame(data = {
        "query": [query],
        "retrieved_doc": [retrieved_doc],
        "llm_answer": [answer],
        "timestamp": [timestamp]
    })

    if os.path.exists(log_file):
        new_data.to_csv(log_file, mode="a", header=False, index=False)
    else:
        new_data.to_csv(log_file, index=False)
        
    # return response
    return dcc.Markdown(response)
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)  
