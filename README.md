# final-project-template
Dependencies:

PySpark
Python 3.x
Libraries for plotting (Matplotlib, Seaborn)
How to Run:

Step 1: Place the CSV file in the designated data directory.
        
        using code: 
    
    here = os.path.abspath('Utility_Energy_Registry_Monthly_ZIP_Code_Energy_Use__2016-2021_20250220.csv')
    
    input_dir = os.path.abspath(os.path.join(here, os.pardir))
    
    data_path = os.path.join(input_dir, "Utility_Energy_Registry_Monthly_ZIP_Code_Energy_Use__2016-2021_20250220.csv")
    
    df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(data_path)

Step 2: Open your terminal or IDE configured for PySpark.

Step 3: Run the Python script 

Step 4: Output files  and plots will be saved in the output folder.

dataset link: https://data.ny.gov/Energy-Environment/Utility-Energy-Registry-Monthly-ZIP-Code-Energy-Us/tzb9-c2c6/about_data (please copy this link to the web browser and open it. It can not be opened by just clicking it and I don't know how.)
