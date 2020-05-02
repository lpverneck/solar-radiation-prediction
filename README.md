# solar-radiation-prediction
( ... )

## Folder structure

    solar-radiation-prediction/
    │
    ├── data/                          # Solar radiation data
    │   ├── raw/                       # Raw data
    │   └── splitted/                  # Separate data according to each station for final validation
    │
    ├── graphic_results/               # Some graphics results
    │
	├── models/                        # Best trained models
	│
	├── notebooks/                     # Some exploratory analysis
	│
	├── tasks/                         # Independent tasks
	│   ├── correlation_graphic.py     # Displays a correlation matrix graphically between the variables
	│   ├── data_split.py              # Split data per station for validation
	│   ├── model_rebuild.py           # Reloads a previously trained model
	│   ├── modeling_analysis.py       # Generation of reports and final graphics
	│   └── stat_script.py             # Statistical tests (To be REMOVED !)
	│   
    ├── base.py                        # Main code responsible for training and optimizing the models
	│
	├── environment.yaml               # Conda environment dependencies
	│ 
	├── .gitignore      
	└── README.md