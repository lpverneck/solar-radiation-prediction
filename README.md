# solar-radiation-prediction
( ... )

## Folder structure

    solar-radiation-prediction/
    │
    ├── data/
    │   ├── processed/                 # data after processing
    │   └── raw/                       # raw data
    │
	├── docs/                          # documentation
	│
	├── models/                        # trained models for each station
	│   ├── w1_models/                 # without features selection
	│	└── w2_models/                 # with features selection
	│
	├── notebooks/
	│   ├── cover.png
	│   ├── eda.ipynb                  # exploratory data analysis
	│   ├── norm_test.ipynb            # normality test
	│	└── norm_test.py               # normality test function
	│
	├── results/                       # general results
	│	└── graphic_results/   
	│
	├── src/                           # Independent tasks
	│   ├── __init__.py
	│   ├── correlation_graphic.py      
	│   ├── data_split.py              # Split data per station for validation
	│   ├── model_rebuild.py
	│   ├── modeling_analysis.py
	│   └── tasks.py
	│   
    ├── main.py                        # Main code
	├── params.py                      # Params settings
	├── poetry.lock
	├── pyproject.toml                 # Environment dependencies
	│ 
	├── .gitignore
	└── README.md