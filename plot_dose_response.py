import pandas as pd

# Specify the path to your Excel file
excel_file = 'C:/GSC Project/drug_dose_response/raw_data/temporarySkanitExport Jun Mon 10 2024 13-31-42-7611086.xlsx'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(excel_file)

# Print the DataFrame to verify the import
print(df)

# Print the skax file name
print(df.iloc[0, 0])

# Subset the DataFrame to include only the plate readings
plate_only_df = df.iloc[8:17, 0:]

# Remove wells with no data
plate_only_df = plate_only_df.drop(columns=[plate_only_df.columns[1], plate_only_df.columns[-1]])
plate_only_df = plate_only_df.dropna()

# Make first row the column names
plate_only_df.columns = plate_only_df.iloc[0]
plate_only_df = plate_only_df[1:]

# Make the first column the row names
plate_only_df = plate_only_df.set_index(plate_only_df.columns[0])

# Print the modified DataFrame
print(plate_only_df)

# Calculate the mean of each column
mean_abs = plate_only_df.mean(axis=0)

# Replace 0 dose with 1e-9 for plotting
concentrations = [1141, 570.5, 285.2, 142.6, 71.3, 35.7, 17.8, 8.9, 4.5, 0]

# Plot the dose-response curve
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Example dose-response function (logistic function)
def dose_response(x, bottom, top, ic50, hill_slope):
    return bottom + (top - bottom) / (1 + (x / ic50)**hill_slope)

# Example data (doses and responses)
doses = np.array(concentrations[:-1])
responses = np.array(mean_abs[:-1])
# Fit the model to the data
popt, pcov = curve_fit(dose_response, doses, responses, bounds=([0, 0, 0, 0], [100, 100, 100, 2]))

# Extract the fitted parameters
bottom, top, ic50, hill_slope = popt
print(f"IC50: {ic50}")

# Generate doses for plotting the fit
# Use a range that covers all data points
doses_fit = np.logspace(np.log10(min(doses)), np.log10(max(doses)), 100)
responses_fit = dose_response(doses_fit, *popt)

# Plot the data and the fit
plt.figure(figsize=(8, 6))
plt.scatter(doses, responses, label='Data', color='blue')
plt.plot(doses_fit, responses_fit, label='Fit', color='red')
plt.axvline(x=ic50, color='green', linestyle='--', label=f'IC50: {ic50:.2f}')
plt.xscale('log')
plt.xlabel('Dose (uM)')
plt.ylabel('Absorbance (Cell Viability)')
plt.legend()
plt.title('Dose-Response Curve')
plt.show()


