# PostgreSQL helper
import pg_helper
# Pandas package
import pandas as pd
# Dash package
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
# Plotly graph and express package
import plotly.graph_objects as go
import plotly.express as px

# Set Website title
app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}], title='Pangco Jaya Sdn Bhd Sales Forecast Dashboard'
)
server = app.server
app.config['suppress_callback_exceptions'] = True

# Parse json dataframe from table and join them together to create a continuous
# Time series graph
def merge_timeplot(dftable):
    dftrain = pd.read_json(dftable['train_df'].tolist()[0])
    dfvalidate = pd.read_json(dftable['validation_df'].tolist()[0])
    dfvalidate.drop('y', axis=1, inplace=True)
    df_con = pd.concat([dftrain, dfvalidate], axis=1)
    dfforecast = pd.read_json(dftable['forecast_df'].tolist()[0])
    framesall = [df_con, dfforecast]
    result_new = pd.concat(framesall)

    model_best = dftable['mape_model'].tolist()[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result_new.index,
                             y=result_new.y, name='Amount'))
    fig.add_trace(go.Scatter(x=result_new.index,
                             y=result_new.Baseline, name='Baseline'))
    if model_best == 'SARIMAX':
        fig.add_trace(go.Scatter(x=result_new.index,
                                  y=result_new.SARIMAX, name='SARIMAX'))
    else:
        if 'SARIMAX' in result_new.columns:
            fig.add_trace(go.Scatter(x=result_new.index,
                                  y=result_new.SARIMAX, name='SARIMAX', visible='legendonly'))
    if model_best == 'Prophet':
        fig.add_trace(go.Scatter(x=result_new.index,
                                 y=result_new.Prophet, name='Prophet'))
    else:
        if 'Prophet' in result_new.columns:
            fig.add_trace(go.Scatter(x=result_new.index,
                                 y=result_new.Prophet, name='Prophet', visible='legendonly'))
    if model_best == 'HWExSmooth':
        fig.add_trace(go.Scatter(x=result_new.index,
                                 y=result_new.HWExSmooth, name='HWExSmooth'))
    else:
        if 'HWExSmooth' in result_new.columns:
            fig.add_trace(go.Scatter(x=result_new.index, y=result_new.HWExSmooth,
                                 name='HWExSmooth', visible='legendonly'))
    if model_best == 'XGBoost':
        fig.add_trace(go.Scatter(x=result_new.index,
                                 y=result_new.XGBoost, name='XGBoost'))
    else:
        if 'XGBoost' in result_new.columns:
            fig.add_trace(go.Scatter(x=result_new.index,
                                 y=result_new.XGBoost, name='XGBoost', visible='legendonly'))
    # Add vertical dash line at the last test month
    fig.add_vline(x=df_con.index.max(), line_width=3,
                  line_dash="dash", line_color='red')
    fig.update_yaxes(visible=True, showticklabels=True, title='Sales')
    fig.update_xaxes(visible=True, showticklabels=True, title='Date')
    return fig

# Filter according to the parameter set by user, then create graph object
def df_filtering(workingdf, tdebtorcode, tcategory, tprojno, tmonth_year):
    df_locdebtor = pd.DataFrame().reindex(columns=workingdf.columns)
    df_loccategory = pd.DataFrame().reindex(columns=workingdf.columns)

    # Filter by Company
    if ((tdebtorcode == "All") & (tcategory == "All") & (tprojno == "All")):
        df_fullpj = workingdf.loc[(workingdf.debtorcode == tdebtorcode) & (
            workingdf.category == tcategory) & (workingdf.projno == tprojno) & (workingdf.last_trainmy == tmonth_year)]
        df_locdebtor = workingdf.loc[(workingdf.debtorcode != 'All') & (
            workingdf.category == 'All') & (workingdf.last_trainmy == tmonth_year)]
        df_loccategory = workingdf.loc[(workingdf.category != 'All') & (
            workingdf.debtorcode == 'All') & (workingdf.last_trainmy == tmonth_year)]
    # Filter by Product line
    elif ((tdebtorcode == "All") & (tcategory == "All") & (tprojno != "")):
        df_fullpj = workingdf.loc[(workingdf.debtorcode == tdebtorcode) & (
            workingdf.category == tcategory) & (workingdf.projno == tprojno) & (workingdf.last_trainmy == tmonth_year)]
        df_locdebtor = workingdf.loc[(workingdf.debtorcode != 'All') & (
            workingdf.category == 'All') & (workingdf.projno == tprojno) & (workingdf.last_trainmy == tmonth_year)]
        df_loccategory = workingdf.loc[(workingdf.category != 'All') & (
            workingdf.debtorcode == 'All') & (workingdf.projno == tprojno) & (workingdf.last_trainmy == tmonth_year)]
        # Filter by Debtor
    elif ((tdebtorcode != "") & (tcategory == "All")):
        df_fullpj = workingdf.loc[(workingdf.debtorcode == tdebtorcode) & (
            workingdf.category == 'All') & (workingdf.last_trainmy == tmonth_year)]
        df_loccategory = workingdf.loc[(workingdf.debtorcode == tdebtorcode) & (
            workingdf.category != 'All') & (workingdf.last_trainmy == tmonth_year)]
        # Filter by Category
    elif ((tdebtorcode == "All") & (tcategory != "")):
        df_fullpj = workingdf.loc[(workingdf.debtorcode == 'All') & (
            workingdf.category == tcategory) & (workingdf.last_trainmy == tmonth_year)]
        df_locdebtor = workingdf.loc[(workingdf.debtorcode != 'All') & (
            workingdf.category == tcategory) & (workingdf.last_trainmy == tmonth_year)]
        # Filter by Debtor + Category
    elif ((tdebtorcode != "") & (tcategory != "")):
        df_fullpj = workingdf.loc[(workingdf.debtorcode == tdebtorcode) & (
            workingdf.category == tcategory) & (workingdf.last_trainmy == tmonth_year)]
    # Return filtered Debtor dataframe for top 10 debtor graph
    sortdebtor_df = df_locdebtor.sort_values(
        by=['forecast_amt'], ascending=False)
    # Return filtered Debtor dataframe for top 10 category graph
    sortcategory_df = df_loccategory.sort_values(
        by=['forecast_amt'], ascending=False)

    # Perform time series graph object creation if filter return with data
    if len(df_fullpj) > 0:
        fig = merge_timeplot(df_fullpj)
    else:
        fig = None
    return fig, df_fullpj, sortdebtor_df, sortcategory_df

# Connection parameters for PostgreSQL DataWarehouse
param_dic = {
    "host": "localhost",
    "database": "datawarehouse",
    "user": "postgres",
    "password": "xxxxxx"
}
# Connect to Data warehouse
conn = pg_helper.postgresql_connect(param_dic)
# Dataframe column header
allcolumn_names = ["debtorcode", "category", "projno", "mape_model", "mape_min", "rmape", "maape_model", "maape_min", "mae_v", "test_mean", "metrics_df", "train_df", "validation_df", "forecast_df", "baseline_df", "last_update", "last_trainmy", "rbaseline", "pred_bt_base", "pred_bt_base_p", "tmonth_amt", "forecast_amt", "pred_trend", "pred_trend_p"]
# SQL String to select from sales_forecast table
sql_string = "SELECT debtorcode, category, projno, mape_model, mape_min, rmape, maape_model, maape_min, mae_v, test_mean, metrics_df, train_df, validation_df, forecast_df, baseline_df, last_update, last_trainmy, rbaseline, pred_bt_base, pred_bt_base_p, tmonth_amt, forecast_amt, pred_trend, pred_trend_p FROM sales_forecast"
# Execute the query
df_alldata = pg_helper.postgresql_to_dataframe(
    conn, sql_string, allcolumn_names)
# Close Data warehouse connection
conn.close()

# Set option and value for debtor filter
debtorslist = df_alldata.debtorcode.unique().tolist()
# Set option and value for category filter
categorylist = df_alldata.category.unique().tolist()
# Set option and value for month year filter
monthyearlist = df_alldata.last_trainmy.unique().tolist()
monthyearmax = max(monthyearlist)

# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                # Company logo
                                html.Img(
                                    src=app.get_asset_url("logo.png"),
                                    id="company-logo",
                                    style={
                                        "height": "40px",
                                        "width": "auto"
                                    },
                                )
                            ],
                        ),

                        html.Div(
                            [
                                # Product line filter
                                html.P(
                                    "Product line",
                                    className="control_label",
                                ),
                                dcc.RadioItems(
                                    id='dw_productline',
                                    options=[{
                                        'label': 'All',
                                        'value': 'All'
                                    },
                                        {
                                        'label': 'UL',
                                        'value': 'UL'
                                    },
                                        {
                                        'label': 'AC',
                                        'value': 'AC'
                                    }
                                    ],
                                    value='All',
                                    className="dcc_control",
                                    labelStyle={"display": "inline-block"},
                                ),
                            ],
                            className="card_container",
                        ),
                        # Debtor filter
                        html.P(
                            "Debtor",
                            className="control_label"),
                        dcc.Dropdown(
                            id='dw_debtor',
                            options=[{'label': c, 'value': c}
                                     for c in debtorslist],
                            value='All',
                            className="dcc_control",
                            clearable=False
                        ),
                        # Category filter
                        html.P(
                            "Category",
                            className="control_label"),
                        dcc.Dropdown(
                            id='dw_category',
                            options=[{'label': c, 'value': c}
                                     for c in categorylist],
                            value='All',
                            className="dcc_control",
                            clearable=False
                        ),
                        # Month Year filter
                        html.P(
                            "Month Year",
                            className="control_label"),
                        dcc.Dropdown(
                            id='dw_month_year',
                            options=[{'label': c, 'value': c}
                                     for c in monthyearlist],
                            value=monthyearmax,
                            className="dcc_control",
                            clearable=False
                        ),
                        # Trend filter
                        html.P(
                            "Trend View",
                            className="control_label",
                        ),
                        dcc.RadioItems(
                            id='dw_trend',
                            options=[{
                                'label': 'All',
                                'value': 'All'
                            },
                                {
                                'label': 'UP',
                                'value': 'UP'
                            },
                                {
                                'label': 'Down',
                                'value': 'DW'
                            },
                                {
                                'label': 'NC',
                                'value': 'NC'
                            }
                            ],
                            value='All',
                            className="dcc_control",
                            labelStyle={"display": "inline-block"},
                        ),
                    ],
                    className="pretty_container three columns",
                    style={'margin-right': '5px'},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H5(id="trend_v", children=['']),
                                        html.P(id="trend_m",
                                               children=["Trend"],
                                               style={'font-weight': 'Bold'}),
                                        html.Div(
                                            [html.I(
                                                className="fas fa-arrows-alt-v")],
                                            className="icon"
                                        )
                                    ],
                                    id="div_trend",
                                    className="mini_container whitefont",
                                ),
                                html.Div(
                                    [
                                        html.H5(id="forecast_v",
                                                children=['']),
                                        html.Div(
                                            [
                                                html.Span(children=['MAE:']),
                                                html.Span(
                                                    id="forecast_m",
                                                    children=['Forecast Err'],
                                                    style={
                                                        'font-weight': 'Bold'}
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            [html.I(
                                                className="fas fa-chart-line")],
                                            className="icon"
                                        )
                                    ],
                                    id="div_forecast",
                                    className="mini_container whitefont",
                                ),
                                html.Div(
                                    [
                                        html.H5(id="baseline_v",
                                                children=['']),
                                        html.P(id="baseline_m", children=[
                                               'than Baseline'], style={'font-weight': 'Bold'}),
                                        html.Div(
                                            [html.I(
                                                className="fas fa-check-circle")],
                                            className="icon"
                                        )
                                    ],
                                    id="div_baseline",
                                    className="mini_container whitefont",
                                ),
                                html.Div(
                                    [
                                        html.H5(id="model_v", children=['']),
                                        html.P(id="model_m", children=[
                                               'Best model'], style={'font-weight': 'Bold'}),
                                        html.Div(
                                            [html.I(
                                                className="fas fa-thumbs-up")],
                                            className="icon"
                                        )
                                    ],
                                    id="div_model",
                                    className="mini_container whitefont",
                                ),
                            ],
                            id="info-container",
                            className="row container-display",
                        ),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="time_series",
                                    config={'displayModeBar': False},
                                )
                            ],
                            className="pretty_container",
                            style={'padding': '0px', 'margin-top': '0px'}
                        ),
                    ],
                    id="right-column",
                    className="ten columns",
                    style={'margin-right': '5px'},
                ),
            ],
            className="row flex-display",
            style={'height': '550px'}
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(
                        id='fig1',
                        config={'displayModeBar': False},
                    )],
                    className="pretty_container six columns",
                    style={'padding': '0px',
                           'margin-top': '0px'}
                ),
                html.Div(
                    [dcc.Graph(
                        id='fig2',
                        config={'displayModeBar': False},
                    )],
                    className="pretty_container six columns",
                    style={'padding': '0px',
                           'margin-top': '0px'}
                ),
            ],
            className="row container-display",
        ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


@app.callback([Output('time_series', 'figure'),
               Output('trend_v', 'children'),
               Output('trend_m', 'children'),
               Output('forecast_v', 'children'),
               Output('forecast_m', 'children'),
               Output('baseline_v', 'children'),
               Output('baseline_m', 'children'),
               Output('model_v', 'children'),
               Output('model_m', 'children'),
               Output('fig1', 'figure'),
               Output('fig2', 'figure')],
              [Input('dw_productline', 'value'),
               Input('dw_debtor', 'value'),
               Input('dw_category', 'value'),
               Input('dw_month_year', 'value'),
              Input('dw_trend', 'value')])
def update_timeseries(dw_productlinev, dw_debtorv, dw_categoryv, dw_month_yearv, trend_valuev):
    if ((dw_debtorv != 'All') & (dw_categoryv != '')):
        fig, workdf, sortdebtor_dfv, sortcategory_dfv = df_filtering(
            df_alldata, dw_debtorv, dw_categoryv, '', dw_month_yearv)
    else:
        fig, workdf, sortdebtor_dfv, sortcategory_dfv = df_filtering(
            df_alldata, dw_debtorv, dw_categoryv, dw_productlinev, dw_month_yearv)

    trend_m = '{:,.2f}'.format(workdf['tmonth_amt'].tolist()[0])
    trend_mp = '{:,.2f}'.format(workdf['pred_trend_p'].tolist()[0])
    trend_v = workdf['pred_trend'].values[0]
    if trend_v == 'NC':
        trend_v = 'No Change'
        trend_ms = trend_m + ' ± ' + trend_mp + '%'
    elif trend_v == 'UP':
        trend_v = 'Up Trend'
        trend_ms = trend_m + ' + ' + trend_mp + '%'
    else:
        trend_v = 'Down Trend'
        trend_ms = trend_m + ' - ' + trend_mp + '%'

    pred_bt_v = workdf['pred_bt_base']
    predbtp = '{:,.2f}'.format(workdf['pred_bt_base_p'].tolist()[0])
    pred_bt_p_v = 'than Baseline: ' + predbtp + "%"
    model_v = workdf['mape_model']
    rmape = '{:,.2f}'.format(workdf['rmape'].tolist()[0])
    model_m = 'Best model: ' + rmape + "%"
    mae_v = '{:,.2f}'.format(workdf['mae_v'].tolist()[0])
    forecast_str = '{:,.2f}'.format(workdf['forecast_amt'].tolist()[0])
    # dfforecast = pd.read_json(workdf['forecast_df'].tolist()[0])
    # forecast_v = dfforecast[model_v].values[0]
    # forecast_str = '{:,.2f}'.format(forecast_v[0])
    forecast_m = " ± " + mae_v
    fig.update_layout(margin=dict(l=70, r=30, t=50, b=50))
    fig.update_layout(title_text='Time series forecast', title_x=0.5)
    fig.update_layout(height=430)

    if trend_valuev == 'UP':
        trend_str = 'UP Trend'
    elif trend_valuev == 'DW':
        trend_str = 'Down Trend'
    elif trend_valuev == 'NC':
        trend_str = 'No Change'
    else:
        trend_str = 'All'

    if sortdebtor_dfv.empty:
        fig1 = go.Figure(data=[go.Scatter(x=[], y=[])])
    elif trend_valuev == 'All':
        sortdebtor_dfv = sortdebtor_dfv.head(10)
    else:
        sortdebtor_dfv = sortdebtor_dfv.loc[(sortdebtor_dfv.pred_trend == trend_valuev)].head(10)

    if sortdebtor_dfv.empty:
        fig1 = go.Figure(data=[go.Scatter(x=[], y=[])])
    else:
        fig1 = go.Figure(data=[go.Bar(
            name='Forecast',
            x=sortdebtor_dfv.debtorcode,
            y=sortdebtor_dfv.forecast_amt
        ),
            go.Bar(
            name='LastMonth',
            x=sortdebtor_dfv.debtorcode,
            y=sortdebtor_dfv.tmonth_amt
        )
        ])
        # fig1 = px.bar(sortdebtor_dfv, x=sortdebtor_dfv.debtorcode,
        #               y=sortdebtor_dfv.forecast_amt, color=sortdebtor_dfv.debtorcode, barmode = 'group')
        fig1.update_yaxes(visible=True, showticklabels=True,
                          title='Sales Forecast')
        fig1.update_xaxes(visible=True, showticklabels=True, title='Debtor')
        fig1.update_layout(margin=dict(l=70, r=30, t=50, b=50))
        fig1.update_layout(
            title_text='Sales Forecast of Top 10 Debtor - ' + trend_str, title_x=0.5)
        fig1.update_layout(height=430)

    if sortcategory_dfv.empty:
        fig2 = go.Figure(data=[go.Scatter(x=[], y=[])])
    elif trend_valuev == 'All':
        sortcategory_dfv = sortcategory_dfv.head(10)
    else:
        sortcategory_dfv = sortcategory_dfv.loc[(sortcategory_dfv.pred_trend == trend_valuev)].head(10)

    if sortcategory_dfv.empty:
        fig2 = go.Figure(data=[go.Scatter(x=[], y=[])])
    else:
        fig2 = go.Figure(data=[go.Bar(
            name='Forecast',
            x=sortcategory_dfv.category,
            y=sortcategory_dfv.forecast_amt
        ),
            go.Bar(
            name='LastMonth',
            x=sortcategory_dfv.category,
            y=sortcategory_dfv.tmonth_amt
        )
        ])
        # fig2 = px.bar(sortcategory_dfv, x=sortcategory_dfv.category,
        #               y=sortcategory_dfv.forecast_amt, color='category')
        fig2.update_yaxes(visible=True, showticklabels=True,
                          title='Sales Forecast')
        fig2.update_xaxes(visible=True, showticklabels=True, title='Category')
        fig2.update_layout(margin=dict(l=70, r=30, t=50, b=50))
        fig2.update_layout(
            title_text='Sales Forecast of Top 10 Category - ' + trend_str, title_x=0.5)
        fig2.update_layout(height=430)

    return fig, trend_v, trend_ms, forecast_str, forecast_m, pred_bt_v, pred_bt_p_v, model_v, model_m, fig1, fig2


@app.callback([Output('dw_debtor', 'options'),
               Output('dw_debtor', 'value')],
              [Input('dw_productline', 'value'),
               Input('fig1', 'clickData'),
               Input('dw_month_year','value')],
              State('dw_debtor','value'))
def update_debtor(dw_productlinev, clickData, dw_month_yearv, dw_debtorsv):
    strout = 'All'
    deblist = [{'label': c, 'value': c} for c in debtorslist]

    ctx = dash.callback_context
    if ctx.triggered:
        input_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if input_id == "fig1":
            if clickData is not None:
                strout = str(clickData['points'][0]['x'])
        else:
            if dw_productlinev != 'All':
                df_fzpj = df_alldata.loc[(df_alldata.projno == dw_productlinev) & (df_alldata.last_trainmy == dw_month_yearv)]
                debtorslistz = df_fzpj.debtorcode.unique().tolist()
            else:
                df_fzpj = df_alldata.loc[(df_alldata.last_trainmy == dw_month_yearv)]
                debtorslistz = df_fzpj.debtorcode.unique().tolist()

            deblist = [{'label': c, 'value': c} for c in debtorslistz]

            if input_id == "dw_month_year":
                if dw_debtorsv in str(deblist):
                    strout = dw_debtorsv
    return deblist, strout


@app.callback([Output('dw_category', 'options'),
               Output('dw_category', 'value')],
              [Input('dw_productline', 'value'),
               Input('dw_debtor', 'value'),
               Input('fig2', 'clickData'),
               Input('dw_month_year','value')],
              State('dw_category','value'))
def update_category(dw_productlinev, dw_debtorv, clickData, dw_month_yearv, dw_categorysv):
    strout = 'All'
    catlist = [{'label': c, 'value': c} for c in categorylist]

    ctx = dash.callback_context
    if ctx.triggered:
        input_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if input_id == "fig2":
            if clickData is not None:
                strout = str(clickData['points'][0]['x'])
        else:
            if dw_debtorv != 'All':
                df_fzpj = df_alldata.loc[(df_alldata.debtorcode == dw_debtorv) & (df_alldata.last_trainmy == dw_month_yearv)]
                categorylistz = df_fzpj.category.unique().tolist()
            elif dw_productlinev != 'All':
                df_fzpj = df_alldata.loc[(df_alldata.projno == dw_productlinev) & (df_alldata.last_trainmy == dw_month_yearv)]
                categorylistz = df_fzpj.category.unique().tolist()
            else:
                df_fzpj = df_alldata.loc[(df_alldata.last_trainmy == dw_month_yearv)]
                categorylistz = df_fzpj.category.unique().tolist()
            catlist = [{'label': c, 'value': c} for c in categorylistz]

            if input_id == "dw_month_year":
                if dw_categorysv in str(catlist):
                    strout = dw_categorysv

    return catlist, strout


# Main
if __name__ == '__main__':
    app.run_server(debug=True)