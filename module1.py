import concurrent.futures
import numpy as np
import firebase_admin
from scipy.optimize import fmin
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
from firebase_admin import credentials, db
from datetime import datetime

cred = credentials.Certificate(r"K:\b\omar\study\grad-bba94-firebase-adminsdk-u9hz8-2c95f70edd.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://grad-bba94-default-rtdb.firebaseio.com/'  # Replace with your actual database URL
})

# Obtain a reference to the node 'yourconstant_key_here'
parent_ref = db.reference('yourconstant_key_here')
# Get the value of the child node
historyFrom = parent_ref.child('historyFrom').get()
historyTo = parent_ref.child('historyTo').get()
selectedPlant = parent_ref.child('selectedPlant').get()
selectedSoilType = parent_ref.child('selectedSoilType').get()
selectedPlantType = parent_ref.child('selectedPlantType').get()
temperatureMin = float(parent_ref.child('temperatureMin').get())
temperatureMax = float(parent_ref.child('temperatureMax').get())
precipitation = float(parent_ref.child('precipitation').get().replace('o', '0'))
Humidity = float(db.reference('Humidity').get())
month1, day1 = historyFrom.split("/")
month2, day2 = historyTo.split("/")


# Function to get Eto value based on the month
def get_eto_value(month):
    eto_values = {
        1: 1.78,   # January
        2: 2.24,   # February
        3: 2.99,   # March
        4: 3.64,   # April
        5: 4.14,   # May
        6: 5.07,   # June
        7: 4.83,   # July
        8: 4.84,   # August
        9: 4.39,   # September
        10: 3.35,  # October
        11: 2.42,  # November
        12: 1.86,  # December
    }
    return eto_values.get(month, 0)  # Return 0 if month not found


# Function to write climate data to a file
def write_climate_data(historyFrom, historyTo, temperatureMin, temperatureMax, precipitation, Humidity):
    try:
        # Validate input data
        if not historyFrom or not historyTo or temperatureMin is None or temperatureMax is None or Humidity is None:
            print("Invalid data retrieved from Firebase.")
            return

        from_date = datetime.strptime(historyFrom, "%m/%d")
        to_date = datetime.strptime(historyTo, "%m/%d")
        month = from_date.month
        eto_value = get_eto_value(month)
        file_path = r"C:\Users\yacout\AppData\Local\Programs\Python\Python310\Lib\site-packages\aquacrop\data\alex_climate.txt"

        with open(file_path, "w") as file:
            file.write("Day\tMonth\tYear\tTmin(C)\tTmax(C)\tPrcp(mm)\tEt0(mm)\n")

        with open(file_path, "a") as file:
            for day in range(from_date.day, to_date.day + 1):
                data_string = f"{day}\t{month}\t2024\t{temperatureMin}\t{temperatureMax}\t{precipitation}\t{eto_value}\n"
                file.write(data_string)
        print("Climate data written successfully.")
    except Exception as e:
        print(f"Error writing climate data: {e}")


# Write the data to the file (only once)
write_climate_data(historyFrom, historyTo, temperatureMin, temperatureMax, precipitation, Humidity)

# Now import pyplot
path = get_filepath('alex_climate.txt')
wdf = prepare_weather(path)


def run_model(smts, max_irr_season, year1, year2):
    plant = Crop(selectedPlant, planting_date=historyFrom)  # define crop
    ground = Soil(selectedSoilType)  # define soil
    init_wc = InitialWaterContent(wc_type='Pct', value=[Humidity])  # define initial soil water conditions
    # define irrigation management
    irrmngt = IrrigationManagement(irrigation_method=1, SMT=smts, MaxIrrSeason=max_irr_season)
    # create and run model
    model = AquaCropModel(f'{year1}/{historyFrom}', f'{year2}/{historyTo}', wdf, ground, plant,
                          irrigation_management=irrmngt, initial_water_content=init_wc)

    model.run_model(till_termination=True)
    results = model.get_simulation_results()
    return model, results


def evaluate(smts, max_irr_season, test=False):
    model, out = run_model(smts, max_irr_season, year1=2024, year2=2024)
    yld = out['Yield potential (tonne/ha)'].mean()
    tirr = out['Seasonal irrigation (mm)'].mean()
    reward = yld

    if test:
        return yld, tirr, reward
    else:
        return -reward


def get_starting_point(num_smts, max_irr_season, num_searches, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)  # Set the random seed for consistency

    x0list = np.random.rand(num_searches, num_smts) * 100
    rlist = []
    for xtest in x0list:
        r = evaluate(xtest, max_irr_season)
        rlist.append(r)
    x0 = x0list[np.argmin(rlist)]
    return x0


def optimize(num_smts, max_irr_season, num_searches=100, random_seed=None):
    x0 = get_starting_point(num_smts, max_irr_season, num_searches, random_seed)
    res = fmin(evaluate, x0, disp=0, args=(max_irr_season,))
    smts = np.squeeze(res)

    # Ensure SMT values are within a safe range
    min_safe_smt = 50  # Example threshold value
    average_smt = np.mean(smts)
    if average_smt < min_safe_smt:
        smts = np.full_like(smts, min_safe_smt)  # Set all SMT values to the minimum safe value

    return smts


def optimize_and_collect(num_smts, max_irr_season, num_searches):
    return optimize(num_smts, max_irr_season, num_searches)


def multiple_optimizations(num_optimizations, num_smts, max_irr_season, num_searches=100):
    stage_smts_list = [[] for _ in range(num_smts)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(optimize_and_collect, num_smts, max_irr_season, num_searches) for _ in range(num_optimizations)]
        for future in concurrent.futures.as_completed(futures):
            smts = future.result()
            for i, smt in enumerate(smts):
                stage_smts_list[i].append(smt)

    average_smts = [np.mean(stage) for stage in stage_smts_list]
    return average_smts


def ensure_min_safety(smts, min_safe_smt=50):
    """Ensure all SMT values are above the minimum safety threshold."""
    return [max(smt, min_safe_smt) for smt in smts]


if __name__ == '__main__':
    # Run the multiple optimization process
    average_smts = multiple_optimizations(10, 4, 300)
    print(f"Average SMTs before safety check: {average_smts}")

    # Ensure all SMT values are above the minimum safety threshold
    safe_smts = ensure_min_safety(average_smts)
    print(f"Average SMTs after safety check: {safe_smts}")

    # Plant type mapping (modify as needed)
    plant_type_to_smt_index = {
        "type 1": 0,
        "type 2": 1,
        "type 3": 2,
        "type 4": 3,
    }

    # Retrieve the SMT value for the selected plant type
    if selectedPlantType in plant_type_to_smt_index:
        smt_index = plant_type_to_smt_index[selectedPlantType]
        smt_value = safe_smts[smt_index]
        rounded_smt_value = round(smt_value)  # Round the SMT value to the nearest integer
        smt_ref = db.reference('smts')
        smt_ref.set(rounded_smt_value)
        print(f"Uploaded SMT {rounded_smt_value} for plant type {selectedPlantType} to Firebase!")
    else:
        print(f"Plant type {selectedPlantType} not found in the mapping.")
