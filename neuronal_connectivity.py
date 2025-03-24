from flask import Flask, render_template, request, jsonify
from markupsafe import Markup
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import openai
import traceback
import sqlite3
from pathlib import Path
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

app = Flask(__name__)


openai.api_key = os.environ.get("OPENAI_API_KEY")

class BrainRegionExplainer:
    """the class to provide AI explanations of brain regions for tool tips"""
    
    def __init__(self):
        """create the explainer with a cache db to avoid repetitive API calls"""  # taken from stack overflow
        # create a database to cache my explanations
        db_path = Path('cache/brain_region_explanations.db')
        db_path.parent.mkdir(exist_ok=True)
        
        self.conn = sqlite3.connect(str(db_path))
        self.cursor = self.conn.cursor()
        
        # create Atable if it doesn't exist
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS explanations (
            region_name TEXT PRIMARY KEY,
            explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        self.conn.commit()
    
    def get_region_explanation(self, region_name):
        """get an explanation for a brain region, using cache if its available"""
        # check if we have a cached explanation
        self.cursor.execute("SELECT explanation FROM explanations WHERE region_name=?", (region_name,))
        result = self.cursor.fetchone()
        
        if result:
            return result[0]
        
        # If not in cache, we create(or generate) a new explanation
        explanation = self._generate_explanation(region_name)
        
        # Store in cache
        self.cursor.execute(
            "INSERT OR REPLACE INTO explanations (region_name, explanation) VALUES (?, ?)",
            (region_name, explanation)
        )
        self.conn.commit()
        
        return explanation
    
    def _generate_explanation(self, region_name):
        """Generate an explanation using OpenAI API"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a neuroscience assistant that provides concise explanations of brain regions. Provide only factual, scientific information in 1-2 sentences."},
                    {"role": "user", "content": f"What is the {region_name} in the mouse brain? Explain its function in 1-2 sentences only."}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            explanation = response.choices[0]['message']['content'].strip()

            return explanation
            
        except Exception as e:
            print(f"Error generating explanation for {region_name}: {e}")
            return f"A region in the mouse brain involved in various neural circuits."  # Fallback
    
    def enhance_plotly_hover_template(self, fig, region_column="Brain Region"):
        """helper function for enhacning a plotly graph with brain region explanations in tooltips"""
        if isinstance(fig.data[0], go.Bar):
            #fetch all regions
            regions = []
            for trace in fig.data:
                if hasattr(trace, 'x') and trace.x is not None:
                    regions.extend(trace.x)
                elif hasattr(trace, 'y') and trace.y is not None:
                    regions.extend(trace.y)
            
            # get explanations for all regions
            explanations = {region: self.get_region_explanation(region) for region in set(regions)}
            
            # Update hover template for each trace
            for trace in fig.data:
                if hasattr(trace, 'x') and trace.x is not None:
                    # horizontal bars
                    hover_template = "<b>%{x}</b><br>Projection Strength: %{y:.4f}<br>%{customdata}<extra></extra>"
                    customdata = [explanations.get(region, "") for region in trace.x]
                    trace.customdata = customdata
                    trace.hovertemplate = hover_template
                elif hasattr(trace, 'y') and trace.y is not None:
                    # vertical bars
                    hover_template = "<b>%{y}</b><br>Projection Strength: %{x:.4f}<br>%{customdata}<extra></extra>"
                    customdata = [explanations.get(region, "") for region in trace.y]
                    trace.customdata = customdata
                    trace.hovertemplate = hover_template
        
        return fig

    def close(self):
        """Close the database connection"""
        self.conn.close()


def get_ai_insights(df, source_name, structure_id):
    """  Generate AI insights about the connectivity patterns for a given brain structure.
    
    Arguments:
        df: DataFrame with the connectivity data (brain regions and projection strengths)
        source_name: Name of the source brain structure
        structure_id: ID of the source brain structure
        
    Returns:
        HTML-formatted string with AI-generated insights
    """
    # process the top connections for insights
    top_connections = df.head(5)[['Brain Region', 'Projection Strength']].to_dict('records')
    connections_text = "\n".join([f"- {conn['Brain Region']}: {conn['Projection Strength']:.4f}" 
                                for conn in top_connections])
    
    # now we make prompt
    prompt = f"""
    I'm analyzing neuronal connectivity data from the Allen Mouse Brain Connectivity Atlas.
    
    Source region: {source_name} (ID: {structure_id})
    Top 5 connections (by projection strength):
    {connections_text}
    
    Based on this data, please provide:
    1. A brief explanation of what these connections might mean functionally
    2. What circuit these connections might be part of
    3. Any relevant research context for this connectivity pattern
    
    Keep your response concise (under 200 words) and focused on helping a neuroscientist understand these connections.
    """
    
    try:
        # call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful neuroscience assistant with expertise in mouse brain connectivity. Provide accurate, concise insights about neuronal connectivity patterns."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350,
            temperature=0.7
        )
        
        # extract and format the response
        insight_text = response.choices[0].message['content'].strip()
        formatted_insight = insight_text.replace('\n\n', '<br><br>').replace('\n', '<br>')

        html_response = f"""
        <div class="card mb-4">
            <div class="card-header bg-primary text-white">
                <h5 class="icon-title mb-0"><i class="bi bi-lightbulb"></i> AI Insights</h5>
            </div>
            <div class="card-body">
                <h6 class="mb-3">Connectivity Analysis for {source_name}</h6>
                <div class="insights-text">
                    {formatted_insight}
                </div>
                <div class="alert alert-info mt-3 mb-0">
                    <small><i class="bi bi-info-circle"></i> These AI-generated insights are meant to assist interpretation 
                    and should be validated against current neuroscience literature.</small>
                </div>
            </div>
        </div>
        """
        
        return Markup(html_response)
        
    except Exception as e:
        # Fallback response if API call fails
        error_html = f"""
        <div class="card mb-4">
            <div class="card-header bg-primary text-white">
                <h5 class="icon-title mb-0"><i class="bi bi-lightbulb"></i> AI Insights</h5>
            </div>
            <div class="card-body">
                <div class="alert alert-warning">
                    <i class="bi bi-exclamation-triangle"></i> Unable to generate insights at this time. 
                    Try exploring the reference section below for information about {source_name} connections.
                </div>
            </div>
        </div>
        """
        print(f"Error generating AI insights: {e}")
        return Markup(error_html)


def handle_brain_query(query, source_name, structure_id, connectivity_data):
    """ Process natural language queries about brain structures and connectivity.
    
    Arguments:
        query: The user's natural language query
        source_name: Name of the source brain structure
        structure_id: ID of the source brain structure
        connectivity_data: DataFrame containing connectivity information
        
    Returns:
        JSON response with the AI's answer
    """
    # convert connectivity data to a string format for the prompt
    connectivity_text = "\n".join([
        f"- {row['Brain Region']}: {row['Projection Strength']:.4f}" 
        for _, row in connectivity_data.head(10).iterrows() # limit to top whatever(10 for now) connections
    ])
    
    # Create a system prompt with context about neuroscience
    system_prompt = """
    You are a specialized AI assistant for the Allen Institute for Brain Science's 
    Neuronal Connectivity Explorer. You help neuroscientists interpret brain connectivity data.
    
    When answering questions:
    1. Be precise, scientific, and accurate
    2. If you don't know, acknowledge limitations
    3. Relate information to functional implications when possible
    4. Keep explanations concise but informative
    5. Focus on data-supported answers rather than speculation
    """
    
    # creating the main prompt with the user's query and context
    main_prompt = f"""
    I'm analyzing the following mouse brain connectivity data:
    
    Source region: {source_name} (ID: {structure_id})
    Connectivity data (projection strengths to other regions):
    {connectivity_text}
    
    My question is: {query}
    """
    
    try:
        # call api
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": main_prompt}
            ],
            max_tokens=500,
            temperature=0.5
        )
        
        # extract the response
        answer = response.choices[0]['message']['content'].strip()
        
        return {
            "status": "success",
            "answer": answer
        }
        
    except Exception as e:
        print(f"Error in AI query: {e}")
        return {
            "status": "error",
            "message": "Sorry, I couldn't process your question at this time."
        }


def fetch_connectivity_data(structure_id):
    """Fetch neuronal connectivity data for a given brain structure ID."""
    try:
        # create the MouseConnectivityCache instance
        mcc = MouseConnectivityCache(manifest_file='connectivity/mouse_connectivity_manifest.json')

        # get all experiments with injections in the specific structure
        experiments = mcc.get_experiments(injection_structure_ids=[structure_id])

        if not experiments:
            print(f"No experiments found for Structure ID {structure_id}")
            return []

        # Use the first valid experiment
        experiment_id = experiments[0]['id']
        print(f"Found experiment ID: {experiment_id}")

        #Get projection matrix     for the specific experiment ID we just founb
        projection_matrix = mcc.get_projection_matrix(experiment_ids=[experiment_id])

        # make sure matrix exists
        if 'matrix' not in projection_matrix or 'rows' not in projection_matrix or 'columns' not in projection_matrix:
            print(f"No valid projection matrix structure found 4 experiment ID {experiment_id}")
            return []

        matrix = projection_matrix['matrix']  #Extract the real numPy array
        rows = projection_matrix['rows']
        columns = projection_matrix['columns']

        
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            print(f"Empty projection matrix for experiment ID {experiment_id}")
            return []

        # get all brain structures
        structure_tree = mcc.get_structure_tree()
        projection_data = []

        # extract data from the projection matrix
        for idx, density in enumerate(matrix[0]):  # use the first row for proj densty
            if density > 0:  # make sure only projects that are valid are included
                target_structure_id = columns[idx]['structure_id']

                target_info = list(structure_tree.get_structures_by_id([target_structure_id]))[0]
                projection_data.append({
                    'structure_name': target_info['name'],
                    'projection_density': float(density) #didnt work normally so I had to cast
                })

        return projection_data

    except Exception as e:
        print(f" Error fetching data: {e}")
        return []



def process_data(projection_data):
    """Process projection data."""
    if not projection_data:
        return [], []
    
    target_regions = [data['structure_name'] for data in projection_data]
    projection_strengths = [data['projection_density'] for data in projection_data]
    
    return target_regions, projection_strengths


def get_example_data():
    """Return example data for testing when API fails. Fallback, of course."""
    return [
        {'structure_name': 'Primary visual area', 'projection_density': 0.85},
        {'structure_name': 'Secondary visual area', 'projection_density': 0.72},
        {'structure_name': 'Thalamus', 'projection_density': 0.65},
        {'structure_name': 'Hypothalamus', 'projection_density': 0.43},
        {'structure_name': 'Cerebellum', 'projection_density': 0.38}
    ]


def create_bar_chart(df, structure_id):
    """Create a bar chart visualization of connectivity data."""
    fig = px.bar(df, x='Brain Region', y='Projection Strength',
                 title=f'Neuronal Connections for Structure ID: {structure_id}',
                 color='Projection Strength',
                 color_continuous_scale='Reds')
                 
    # background color to make light bars more visible
    fig.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='#e6e6e6',  # Light gray background
        paper_bgcolor='#f8f9fa'
    )
    
    # grid lines to be more visible
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#d0d0d0'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#d0d0d0'
    )
    
    # add tooltips with AI explanations
    explainer = BrainRegionExplainer()
    fig = explainer.enhance_plotly_hover_template(fig)
    explainer.close()
    
    fig.update_traces(
        marker_line_width=1.5, 
        opacity=0.9
    )
    fig.update_layout(
    
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial",
        namelength=-1,
        align="left"
    ),
    hovermode="closest"
    )
    
    return fig.to_html(full_html=False)


def create_heatmap(df, structure_id, source_name):
    """Create a heatmap visualization of connectivity data."""
    # making the matrix for the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=df['Projection Strength'].values.reshape(-1, 1),
        y=df['Brain Region'],
        x=[source_name],
        colorscale='Reds',
        hovertemplate="From: %{x}<br>To: %{y}<br>Strength: %{z:.4f}<br>%{customdata}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f'Connectivity Heatmap for {source_name} (ID: {structure_id})',
        height=max(500, len(df) * 25),  #scale height based on # of regions
        margin=dict(l=150),  # + margin for region names
        yaxis=dict(title='Target Brain Region'),
        xaxis=dict(title='Source Brain Region')
    )
    
    # Add enhanced tooltips with AI explanations
    explainer = BrainRegionExplainer()
    fig = explainer.enhance_plotly_hover_template(fig)
    explainer.close()
    
    return fig.to_html(full_html=False)


def create_network_graph(df, source_name, structure_id):
    """Create a network graph visualization of brain connections."""
    # Sort and limit to top connections for clarity
    network_df = df.copy().sort_values('Projection Strength', ascending=False).head(10)
    
    # Calculate positions in a circular layout
    n_nodes = len(network_df) + 1  # +1 for source node
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False).tolist()
    
    # Initializing the node positions
    node_x = [0.5] + [0.5 + 0.4*np.cos(angle) for angle in angles[:-1]]
    node_y = [0.5] + [0.5 + 0.4*np.sin(angle) for angle in angles[:-1]]
    
    # Scale node sizes and normalize for better visibility
    max_strength = network_df['Projection Strength'].max()
    min_strength = network_df['Projection Strength'].min()
    
    
    node_sizes = [50]  # Source node size
    for strength in network_df['Projection Strength']:
        if max_strength == min_strength:
            norm_size = 25  # Default size if all values are the same
        else:
            norm_size = 15 + (strength - min_strength) / (max_strength - min_strength) * 25  # and here is where we scale the nodes
        node_sizes.append(norm_size)
    
    # Node text and their labels
    node_text = [f"{source_name} (source)"] + [f"{region} ({strength:.3f})" 
                                             for region, strength in zip(network_df['Brain Region'], 
                                                                        network_df['Projection Strength'])]
    
    # Create scatter plot for nodes
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=['#D62728'] + ['#1F77B4']*len(network_df),  # Red for source, blue for targets
            line=dict(width=2, color='white')
        ),
        text=node_text,
        textposition="top center",
        hoverinfo='text'
    )
    
    # clear figure first
    fig = go.Figure()
    fig.add_trace(node_trace)
    
    # add edges with scaled widths
    for i, (_, row) in enumerate(network_df.iterrows()):
        # calculate normalized line width between 1 and 8
        if max_strength == min_strength:
            width = 4  # Default width if all values are the same
        else:
            width = 1 + ((row['Projection Strength'] - min_strength) / (max_strength - min_strength)) * 7
        
        # line shape for each connection
        fig.add_shape(
            type="line",
            x0=0.5, y0=0.5,  # Source node
            x1=node_x[i+1], y1=node_y[i+1],  # Target node
            line=dict(
                color='rgba(31, 119, 180, 0.6)',
                width=width
            )
        )
        
        # add annotation with strength value near the middle of each line
        midx = (0.5 + node_x[i+1]) / 2
        midy = (0.5 + node_y[i+1]) / 2
        fig.add_annotation(
            x=midx, y=midy,
            text=f"{row['Projection Strength']:.3f}",
            showarrow=False,
            font=dict(size=9, color="black"),
            bgcolor="rgba(255, 255, 255, 0.7)"
        )
    

    fig.update_layout(
        title=f'Brain Network: Top 10 Projections from {source_name} (ID: {structure_id})',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        margin=dict(l=20, r=20, t=60, b=20),
        height=600,
        width=700,
        plot_bgcolor='#f8f9fa',
        annotations=[
            dict(
                text="Thicker lines = stronger connections",
                x=0.5, y=0.02,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )

    fig.update_layout(
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial",
        namelength=-1,
        align="left"
    ),
    hovermode="closest"
    )
    
    # reproducability is proven here by me using the same method signatures to call for brain region explainer
    explainer = BrainRegionExplainer()
    fig = explainer.enhance_plotly_hover_template(fig)
    explainer.close()
    
    return fig.to_html(full_html=False)


@app.route('/query', methods=['POST'])
def process_query():
    """Process natural language queries about brain structures."""
    data = request.json
    query = data.get('query')
    source_name = data.get('source_name')
    structure_id = data.get('structure_id')
    
    # Retrieve connectivity data for this structure
    try:
        # Try to fetch real data
        projection_data = fetch_connectivity_data(int(structure_id))
        if not projection_data:
            projection_data = get_example_data()
            
        target_regions, projection_strengths = process_data(projection_data)
        df = pd.DataFrame({'Brain Region': target_regions, 
                          'Projection Strength': projection_strengths})
        df = df.sort_values('Projection Strength', ascending=False)
        
        response = handle_brain_query(query, source_name, structure_id, df)
        return jsonify(response)
        
    except Exception as e:
        print("Error in query processing:")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": "An error occurred while processing your query."
        })


@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route for the application."""
    bar_html = None
    heatmap_html = None
    network_html = None
    error_message = None
    source_name = None
    ai_insights_html = None
    structure_id = None
    
    if request.method == 'POST':
        structure_id = request.form.get('structure_id')
        
        if structure_id:
            try:
                structure_id = int(structure_id) 
                projection_data = fetch_connectivity_data(structure_id) 
                
                # attempt to get the source structure Name
                try:
                    mcc = MouseConnectivityCache(manifest_file='connectivity/mouse_connectivity_manifest.json')
                    structure_tree = mcc.get_structure_tree()
                    source_info = list(structure_tree.get_structures_by_id([structure_id]))
                    if source_info:
                        source_name = source_info[0].get('name', f"Structure {structure_id}")
                    else:
                        print(f"No structure info found for ID: {structure_id}")
                        source_name = f"Structure {structure_id}"
                except:
                    source_name = f"Structure {structure_id}"
                
                if not projection_data:
                    print("no data from API, using example data")
                    projection_data = get_example_data()
                    error_message = "No real data found for this structure ID. Showing example data instead."
                
                target_regions, projection_strengths = process_data(projection_data)
                df = pd.DataFrame({'Brain Region': target_regions, 
                                   'Projection Strength': projection_strengths})
                
                #sort this by projection strength
                df = df.sort_values('Projection Strength', ascending=False)
                if len(df) > 20:
                    df = df.head(20)
                
                # Create the three visualizations
                bar_html = create_bar_chart(df, structure_id)
                heatmap_html = create_heatmap(df, structure_id, source_name)
                network_html = create_network_graph(df, source_name, structure_id)
                
                # Generate AI insights
                ai_insights_html = get_ai_insights(df, source_name, structure_id)
            
            except ValueError:
                error_message = "Invalid input... Please enter a numeric structure ID."
            except Exception as e:
                error_message = f"Error processing request: {e}"
                print(f"Exception details: {e}")
                # my fallback to example data
                projection_data = get_example_data()
                target_regions, projection_strengths = process_data(projection_data)
                df = pd.DataFrame({'Brain Region': target_regions, 
                                  'Projection Strength': projection_strengths})
                source_name = "Example Structure"
                structure_id = 0
                bar_html = create_bar_chart(df, 0)
                heatmap_html = create_heatmap(df, 0, source_name)
                network_html = create_network_graph(df, source_name, 0)
    
    return render_template('index.html', 
                          bar_html=bar_html, 
                          heatmap_html=heatmap_html, 
                          network_html=network_html, 
                          error_message=error_message,
                          source_name=source_name,
                          structure_id=structure_id,
                          ai_insights_html=ai_insights_html)


@app.route('/structures')
def get_brain_structures():
    try:
        csv_path = os.path.join(app.root_path, 'static', 'structures.csv')
        df = pd.read_csv(csv_path)
        structure_list = df.to_dict(orient='records')
        return jsonify(structure_list)
    except Exception as e:
        print(f"Failed to load structures from CSV: {e}")
        return jsonify([])

if __name__ == "__main__":
    app.run(debug=True)
