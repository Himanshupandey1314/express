{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1ib8gFj18Qat_bazQtF7dJAoAoYTxQ-Mw",
      "authorship_tag": "ABX9TyPPgfDswpcaTsrWxqV0+z94"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "GRpI-SCVcJkG"
      },
      "outputs": [],
      "source": [
        "pip install streamlit --quiet"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install pyngrok"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "D5FVnedccrTd",
        "outputId": "871e868f-a634-42a7-8a1e-73075d7d96b2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: pyngrok in /usr/local/lib/python3.10/dist-packages (7.2.1)\n",
            "Requirement already satisfied: PyYAML>=5.1 in /usr/local/lib/python3.10/dist-packages (from pyngrok) (6.0.2)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install vectorbt"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "u2y0FISjcvaN",
        "outputId": "032fc144-04ca-4725-afa2-771841de23ad"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: vectorbt in /usr/local/lib/python3.10/dist-packages (0.26.2)\n",
            "Requirement already satisfied: numpy<2.0.0,>=1.16.5 in /usr/local/lib/python3.10/dist-packages (from vectorbt) (1.23.5)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from vectorbt) (2.2.2)\n",
            "Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from vectorbt) (1.13.1)\n",
            "Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (from vectorbt) (3.8.0)\n",
            "Requirement already satisfied: plotly>=4.12.0 in /usr/local/lib/python3.10/dist-packages (from vectorbt) (5.24.1)\n",
            "Requirement already satisfied: ipywidgets>=7.0.0 in /usr/local/lib/python3.10/dist-packages (from vectorbt) (7.7.1)\n",
            "Requirement already satisfied: dill in /usr/local/lib/python3.10/dist-packages (from vectorbt) (0.3.9)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from vectorbt) (4.66.6)\n",
            "Requirement already satisfied: dateparser in /usr/local/lib/python3.10/dist-packages (from vectorbt) (1.2.0)\n",
            "Requirement already satisfied: imageio in /usr/local/lib/python3.10/dist-packages (from vectorbt) (2.36.0)\n",
            "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (from vectorbt) (1.5.2)\n",
            "Requirement already satisfied: schedule in /usr/local/lib/python3.10/dist-packages (from vectorbt) (1.2.2)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from vectorbt) (2.32.3)\n",
            "Requirement already satisfied: pytz in /usr/local/lib/python3.10/dist-packages (from vectorbt) (2024.2)\n",
            "Requirement already satisfied: mypy-extensions in /usr/local/lib/python3.10/dist-packages (from vectorbt) (1.0.0)\n",
            "Requirement already satisfied: numba<0.57.0,>=0.56.0 in /usr/local/lib/python3.10/dist-packages (from vectorbt) (0.56.4)\n",
            "Requirement already satisfied: ipykernel>=4.5.1 in /usr/local/lib/python3.10/dist-packages (from ipywidgets>=7.0.0->vectorbt) (5.5.6)\n",
            "Requirement already satisfied: ipython-genutils~=0.2.0 in /usr/local/lib/python3.10/dist-packages (from ipywidgets>=7.0.0->vectorbt) (0.2.0)\n",
            "Requirement already satisfied: traitlets>=4.3.1 in /usr/local/lib/python3.10/dist-packages (from ipywidgets>=7.0.0->vectorbt) (5.7.1)\n",
            "Requirement already satisfied: widgetsnbextension~=3.6.0 in /usr/local/lib/python3.10/dist-packages (from ipywidgets>=7.0.0->vectorbt) (3.6.10)\n",
            "Requirement already satisfied: ipython>=4.0.0 in /usr/local/lib/python3.10/dist-packages (from ipywidgets>=7.0.0->vectorbt) (7.34.0)\n",
            "Requirement already satisfied: jupyterlab-widgets>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from ipywidgets>=7.0.0->vectorbt) (3.0.13)\n",
            "Requirement already satisfied: llvmlite<0.40,>=0.39.0dev0 in /usr/local/lib/python3.10/dist-packages (from numba<0.57.0,>=0.56.0->vectorbt) (0.39.1)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from numba<0.57.0,>=0.56.0->vectorbt) (75.1.0)\n",
            "Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from plotly>=4.12.0->vectorbt) (9.0.0)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from plotly>=4.12.0->vectorbt) (24.2)\n",
            "Requirement already satisfied: python-dateutil in /usr/local/lib/python3.10/dist-packages (from dateparser->vectorbt) (2.8.2)\n",
            "Requirement already satisfied: regex!=2019.02.19,!=2021.8.27 in /usr/local/lib/python3.10/dist-packages (from dateparser->vectorbt) (2024.9.11)\n",
            "Requirement already satisfied: tzlocal in /usr/local/lib/python3.10/dist-packages (from dateparser->vectorbt) (5.2)\n",
            "Requirement already satisfied: pillow>=8.3.2 in /usr/local/lib/python3.10/dist-packages (from imageio->vectorbt) (11.0.0)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->vectorbt) (1.3.1)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib->vectorbt) (0.12.1)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->vectorbt) (4.54.1)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->vectorbt) (1.4.7)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->vectorbt) (3.2.0)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas->vectorbt) (2024.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->vectorbt) (3.4.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->vectorbt) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->vectorbt) (2.2.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->vectorbt) (2024.8.30)\n",
            "Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->vectorbt) (1.4.2)\n",
            "Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->vectorbt) (3.5.0)\n",
            "Requirement already satisfied: jupyter-client in /usr/local/lib/python3.10/dist-packages (from ipykernel>=4.5.1->ipywidgets>=7.0.0->vectorbt) (6.1.12)\n",
            "Requirement already satisfied: tornado>=4.2 in /usr/local/lib/python3.10/dist-packages (from ipykernel>=4.5.1->ipywidgets>=7.0.0->vectorbt) (6.3.3)\n",
            "Requirement already satisfied: jedi>=0.16 in /usr/local/lib/python3.10/dist-packages (from ipython>=4.0.0->ipywidgets>=7.0.0->vectorbt) (0.19.2)\n",
            "Requirement already satisfied: decorator in /usr/local/lib/python3.10/dist-packages (from ipython>=4.0.0->ipywidgets>=7.0.0->vectorbt) (4.4.2)\n",
            "Requirement already satisfied: pickleshare in /usr/local/lib/python3.10/dist-packages (from ipython>=4.0.0->ipywidgets>=7.0.0->vectorbt) (0.7.5)\n",
            "Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from ipython>=4.0.0->ipywidgets>=7.0.0->vectorbt) (3.0.48)\n",
            "Requirement already satisfied: pygments in /usr/local/lib/python3.10/dist-packages (from ipython>=4.0.0->ipywidgets>=7.0.0->vectorbt) (2.18.0)\n",
            "Requirement already satisfied: backcall in /usr/local/lib/python3.10/dist-packages (from ipython>=4.0.0->ipywidgets>=7.0.0->vectorbt) (0.2.0)\n",
            "Requirement already satisfied: matplotlib-inline in /usr/local/lib/python3.10/dist-packages (from ipython>=4.0.0->ipywidgets>=7.0.0->vectorbt) (0.1.7)\n",
            "Requirement already satisfied: pexpect>4.3 in /usr/local/lib/python3.10/dist-packages (from ipython>=4.0.0->ipywidgets>=7.0.0->vectorbt) (4.9.0)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil->dateparser->vectorbt) (1.16.0)\n",
            "Requirement already satisfied: notebook>=4.4.1 in /usr/local/lib/python3.10/dist-packages (from widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (6.5.5)\n",
            "Requirement already satisfied: parso<0.9.0,>=0.8.4 in /usr/local/lib/python3.10/dist-packages (from jedi>=0.16->ipython>=4.0.0->ipywidgets>=7.0.0->vectorbt) (0.8.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (3.1.4)\n",
            "Requirement already satisfied: pyzmq<25,>=17 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (24.0.1)\n",
            "Requirement already satisfied: argon2-cffi in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (23.1.0)\n",
            "Requirement already satisfied: jupyter-core>=4.6.1 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (5.7.2)\n",
            "Requirement already satisfied: nbformat in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (5.10.4)\n",
            "Requirement already satisfied: nbconvert>=5 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (7.16.4)\n",
            "Requirement already satisfied: nest-asyncio>=1.5 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (1.6.0)\n",
            "Requirement already satisfied: Send2Trash>=1.8.0 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (1.8.3)\n",
            "Requirement already satisfied: terminado>=0.8.3 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (0.18.1)\n",
            "Requirement already satisfied: prometheus-client in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (0.21.0)\n",
            "Requirement already satisfied: nbclassic>=0.4.7 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (1.1.0)\n",
            "Requirement already satisfied: ptyprocess>=0.5 in /usr/local/lib/python3.10/dist-packages (from pexpect>4.3->ipython>=4.0.0->ipywidgets>=7.0.0->vectorbt) (0.7.0)\n",
            "Requirement already satisfied: wcwidth in /usr/local/lib/python3.10/dist-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython>=4.0.0->ipywidgets>=7.0.0->vectorbt) (0.2.13)\n",
            "Requirement already satisfied: platformdirs>=2.5 in /usr/local/lib/python3.10/dist-packages (from jupyter-core>=4.6.1->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (4.3.6)\n",
            "Requirement already satisfied: notebook-shim>=0.2.3 in /usr/local/lib/python3.10/dist-packages (from nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (0.2.4)\n",
            "Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (4.12.3)\n",
            "Requirement already satisfied: bleach!=5.0.0 in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (6.2.0)\n",
            "Requirement already satisfied: defusedxml in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (0.7.1)\n",
            "Requirement already satisfied: jupyterlab-pygments in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (0.3.0)\n",
            "Requirement already satisfied: markupsafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (3.0.2)\n",
            "Requirement already satisfied: mistune<4,>=2.0.3 in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (3.0.2)\n",
            "Requirement already satisfied: nbclient>=0.5.0 in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (0.10.0)\n",
            "Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (1.5.1)\n",
            "Requirement already satisfied: tinycss2 in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (1.4.0)\n",
            "Requirement already satisfied: fastjsonschema>=2.15 in /usr/local/lib/python3.10/dist-packages (from nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (2.20.0)\n",
            "Requirement already satisfied: jsonschema>=2.6 in /usr/local/lib/python3.10/dist-packages (from nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (4.23.0)\n",
            "Requirement already satisfied: argon2-cffi-bindings in /usr/local/lib/python3.10/dist-packages (from argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (21.2.0)\n",
            "Requirement already satisfied: webencodings in /usr/local/lib/python3.10/dist-packages (from bleach!=5.0.0->nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (0.5.1)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (24.2.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (2024.10.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (0.35.1)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (0.21.0)\n",
            "Requirement already satisfied: jupyter-server<3,>=1.8 in /usr/local/lib/python3.10/dist-packages (from notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (1.24.0)\n",
            "Requirement already satisfied: cffi>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from argon2-cffi-bindings->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (1.17.1)\n",
            "Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (2.6)\n",
            "Requirement already satisfied: pycparser in /usr/local/lib/python3.10/dist-packages (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (2.22)\n",
            "Requirement already satisfied: anyio<4,>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (3.7.1)\n",
            "Requirement already satisfied: websocket-client in /usr/local/lib/python3.10/dist-packages (from jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (1.8.0)\n",
            "Requirement already satisfied: sniffio>=1.1 in /usr/local/lib/python3.10/dist-packages (from anyio<4,>=3.1.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (1.3.1)\n",
            "Requirement already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio<4,>=3.1.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->vectorbt) (1.2.2)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile app.py\n",
        "import streamlit as st\n",
        "import vectorbt as vbt\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import plotly.graph_objs as go\n",
        "from datetime import datetime\n",
        "import pytz  # Make sure pytz is installed\n",
        "\n",
        "# Convert date to datetime with timezone\n",
        "def convert_to_timezone_aware(date_obj):\n",
        "    return datetime.combine(date_obj, datetime.min.time()).replace(tzinfo=pytz.UTC)\n",
        "\n",
        "# Streamlit interface\n",
        "\n",
        "st.set_page_config(page_title='Backtesting engine', layout='wide')\n",
        "st.title(\"Stock AI backtesting engine\")\n",
        "\n",
        "# Sidebar for inputs\n",
        "with st.sidebar:\n",
        "    # Inputs for the symbol, start and end dates\n",
        "    st.header(\"Strategy Controls\")\n",
        "\n",
        "    # Inputs for the symbol, start and end dates\n",
        "    symbol = st.text_input(\"Enter the symbol (e.g., 'AAPL')\", value=\"HDFCBANK.NS\")\n",
        "    start_date = st.date_input(\"Start Date\", value=pd.to_datetime(\"2010-01-01\"))\n",
        "    end_date = st.date_input(\"End Date\", value=pd.to_datetime(\"2023-01-01\"))\n",
        "\n",
        "    # EMA controls\n",
        "    short_ema_period = st.number_input(\"Short EMA Period\", value=10, min_value=1)\n",
        "    long_ema_period = st.number_input(\"Long EMA Period\", value=20, min_value=1)\n",
        "\n",
        "    st.header(\"Backtesting Controls\")\n",
        "\n",
        "    # Backtesting controls\n",
        "    initial_equity = st.number_input(\"Initial Equity\", value=100000)\n",
        "    size = st.text_input(\"Position Size\", value='50')  # Text input for size\n",
        "    size_type = st.selectbox(\"Size Type\", [\"amount\", \"value\", \"percent\"], index=2)  # Dropdown for size type\n",
        "    fees = st.number_input(\"Fees (as %)\", value=0.12, format=\"%.4f\")\n",
        "    direction = st.selectbox(\"Direction\", [\"longonly\", \"shortonly\", \"both\"], index=0)\n",
        "\n",
        "    # Button to perform backtesting\n",
        "    backtest_clicked = st.button(\"Backtest\")\n",
        "\n",
        "# Main area for results\n",
        "if backtest_clicked:\n",
        "    start_date_tz = convert_to_timezone_aware(start_date)\n",
        "    end_date_tz = convert_to_timezone_aware(end_date)\n",
        "\n",
        "    # Fetch data\n",
        "    data = vbt.YFData.download(symbol, start=start_date_tz, end=end_date_tz).get('Close')\n",
        "\n",
        "    # Calculate EMAs and signals\n",
        "    short_ema = vbt.MA.run(data, short_ema_period, short_name='fast', ewm=True)\n",
        "    long_ema = vbt.MA.run(data, long_ema_period, short_name='slow', ewm=True)\n",
        "    entries = short_ema.ma_crossed_above(long_ema)\n",
        "    exits = short_ema.ma_crossed_below(long_ema)\n",
        "\n",
        "    # Convert size to appropriate type\n",
        "    if size_type == 'percent':\n",
        "        size_value = float(size) / 100.0\n",
        "    else:\n",
        "        size_value = float(size)\n",
        "\n",
        "    # Run portfolio\n",
        "    portfolio = vbt.Portfolio.from_signals(\n",
        "        data, entries, exits,\n",
        "        direction=direction,\n",
        "        size=size_value,\n",
        "        size_type=size_type,\n",
        "        fees=fees/100,\n",
        "        init_cash=initial_equity,\n",
        "        freq='1D',\n",
        "        min_size =1,\n",
        "        size_granularity = 1\n",
        "    )\n",
        "\n",
        "    # Create tabs\n",
        "    tab1, tab2, tab3, tab4, tab5 = st.tabs([\"Backtesting Stats\", \"List of Trades\",\n",
        "                                          \"Equity Curve\", \"Drawdown\", \"Portfolio Plot\"])\n",
        "\n",
        "\n",
        "    with tab1:\n",
        "        # Display results\n",
        "        st.markdown(\"**Backtesting Stats:**\")\n",
        "        stats_df = pd.DataFrame(portfolio.stats(), columns=['Value'])\n",
        "        stats_df.index.name = 'Metric'  # Set the index name to 'Metric' to serve as the header\n",
        "        st.dataframe(stats_df, height=800)  # Adjust the height as needed to remove the scrollbar\n",
        "\n",
        "    with tab2:\n",
        "        st.markdown(\"**List of Trades:**\")\n",
        "        trades_df = portfolio.trades.records_readable\n",
        "        trades_df = trades_df.round(2)  # Rounding the values for better readability\n",
        "        trades_df.index.name = 'Trade No'  # Set the index name to 'Trade Name' to serve as the header\n",
        "        trades_df.drop(trades_df.columns[[0,1]], axis=1, inplace=True)\n",
        "        st.dataframe(trades_df, width=800,height=600)  # Set index to False and use full width\n",
        "\n",
        "\n",
        "    # Plotting\n",
        "    equity_data = portfolio.value()\n",
        "    drawdown_data = portfolio.drawdown() * 100\n",
        "\n",
        "    with tab3:\n",
        "    # Equity Curve\n",
        "        equity_trace = go.Scatter(x=equity_data.index, y=equity_data, mode='lines', name='Equity',line=dict(color='green') )\n",
        "        equity_fig = go.Figure(data=[equity_trace])\n",
        "        equity_fig.update_layout(title='Equity Curve', xaxis_title='Date', yaxis_title='Equity',\n",
        "                                 width=800,height=600)\n",
        "        st.plotly_chart(equity_fig)\n",
        "\n",
        "    with tab4:\n",
        "        # Drawdown Curve\n",
        "        drawdown_trace = go.Scatter(\n",
        "            x=drawdown_data.index,\n",
        "            y=drawdown_data,\n",
        "            mode='lines',\n",
        "            name='Drawdown',\n",
        "            fill='tozeroy',\n",
        "            line=dict(color='red')  # Set the line color to red\n",
        "        )\n",
        "        drawdown_fig = go.Figure(data=[drawdown_trace])\n",
        "        drawdown_fig.update_layout(\n",
        "            title='Drawdown Curve',\n",
        "            xaxis_title='Date',\n",
        "            yaxis_title='% Drawdown',\n",
        "            template='plotly_white',\n",
        "            width = 800,\n",
        "            height = 600\n",
        "        )\n",
        "        st.plotly_chart(drawdown_fig)\n",
        "\n",
        "    with tab5:\n",
        "        # Portfolio Plot\n",
        "        st.markdown(\"**Portfolio Plot:**\")\n",
        "        st.plotly_chart(portfolio.plot())\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Sk5bEyVmc0SA",
        "outputId": "b45046c0-f7e0-45f2-b97f-f82d1cb9f00b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Overwriting app.py\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        " ! curl 4.icanhazip.com"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nlPdIL2gdgv3",
        "outputId": "2a13f7e1-3eb8-45f7-f2dc-258998dea608"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "35.245.222.55\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!npm install -g localtunnel"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Z7-oUaXCdo1k",
        "outputId": "d239b266-7e76-45c6-c050-3720278fe104"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[K\u001b[?25h\n",
            "changed 22 packages, and audited 23 packages in 2s\n",
            "\n",
            "3 packages are looking for funding\n",
            "  run `npm fund` for details\n",
            "\n",
            "1 \u001b[33m\u001b[1mmoderate\u001b[22m\u001b[39m severity vulnerability\n",
            "\n",
            "To address all issues (including breaking changes), run:\n",
            "  npm audit fix --force\n",
            "\n",
            "Run `npm audit` for details.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!streamlit run app.py & npx localtunnel --port 8501"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZL3DguyuduKm",
        "outputId": "92708385-5db1-4db6-edcb-3c5855329735"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.\n",
            "\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Local URL: \u001b[0m\u001b[1mhttp://localhost:8501\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.12:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://35.245.222.55:8501\u001b[0m\n",
            "\u001b[0m\n",
            "your url is: https://sharp-pumas-make.loca.lt\n",
            "2024-11-19 17:57:19.936 Serialization of dataframe to Arrow table was unsuccessful due to: (\"object of type <class 'pandas._libs.tslibs.timedeltas.Timedelta'> cannot be converted to int\", 'Conversion failed for column Value with type object'). Applying automatic fixes for column types to make the dataframe Arrow-compatible.\n",
            "2024-11-19 17:59:02.044 Serialization of dataframe to Arrow table was unsuccessful due to: (\"object of type <class 'pandas._libs.tslibs.timedeltas.Timedelta'> cannot be converted to int\", 'Conversion failed for column Value with type object'). Applying automatic fixes for column types to make the dataframe Arrow-compatible.\n",
            "2024-11-19 18:01:16.143 Serialization of dataframe to Arrow table was unsuccessful due to: (\"object of type <class 'pandas._libs.tslibs.timedeltas.Timedelta'> cannot be converted to int\", 'Conversion failed for column Value with type object'). Applying automatic fixes for column types to make the dataframe Arrow-compatible.\n"
          ]
        }
      ]
    }
  ]
}