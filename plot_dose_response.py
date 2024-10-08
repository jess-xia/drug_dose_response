import pandas as pd
from pathvalidate import sanitize_filename

# Specify the path to your Excel file
# excel_file = 'C:/GSC Project/drug_dose_response/raw_data/temporarySkanitExport Jun Mon 10 2024 13-31-42-7611086.xlsx'
# plot_title = 'G571 Epacadostat (50 min post MTS addition)'

# excel_file = 'C:/GSC Project/drug_dose_response/raw_data/temporarySkanitExport Jun Mon 10 2024 13-32-05-7272353.xlsx'
# plot_title = 'G571 Epacadostat (100 min post MTS addition)'


# excel_file = 'C:/GSC Project/drug_dose_response/raw_data/temporarySkanitExport Jun Mon 10 2024 13-32-25-6989613.xlsx'
# plot_title = 'G571 Epacadostat (180 min post MTS addition)'


# excel_file = 'C:/GSC Project/drug_dose_response/raw_data/temporarySkanitExport Jun Mon 10 2024 13-32-40-4512711.xlsx'
# plot_title = 'G571 5-FU (50 min post MTS addition)'
# plot_title = 'G571 1-MT (50 min post MTS addition)'

# excel_file = 'C:/GSC Project/drug_dose_response/raw_data/temporarySkanitExport Jun Mon 10 2024 13-32-53-5868221.xlsx'
# plot_title = 'G571 5-FU (100 min post MTS addition)'
# plot_title = 'G571 1-MT (100 min post MTS addition)'

# excel_file = 'C:/GSC Project/drug_dose_response/raw_data/temporarySkanitExport Jun Mon 10 2024 13-33-17-3498565.xlsx'
# plot_title = 'G571 5-FU (180 min post MTS addition)'
# plot_title = 'G571 1-MT (180 min post MTS addition)'

excel_file = 'C:/GSC Project/drug_dose_response/raw_data/temporarySkanitExport May Tue 28 2024 14-36-36-7921036.xlsx'
plot_title = 'G571 (20 min post MTS addition)'



plot_filename = sanitize_filename(plot_title)
plot_filename = plot_filename.replace(' ', '_')

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
# mean_abs = plate_only_df.mean(axis=0)
# For when there are no replicates
# 5-FU or Cell number G571
mean_abs = plate_only_df.iloc[0, :]
# 1-MT or cell number HeLa
# mean_abs = plate_only_df.iloc[1, :]




# Replace 0 dose with 1e-9 for plotting
# Epacadostat concentrations
# concentrations = [1141, 570.5, 285.2, 142.6, 71.3, 35.7, 17.8, 8.9, 4.5, 0]
# 5-FU concentrations
# concentrations = [400, 200, 100, 50, 25, 12.5, 6.25, 3.125, 1.0625, 0]
# 1-MT concentrations
# concentrations = [114.6, 57.3, 28.6, 14.3, 7.16, 3.58, 1.79, 0.89, 0.45, 0]


# Replace 0 dose with 1e-9 for plotting
# mean_abs = mean_abs.replace(0, 1e-6)


# Plot the dose-response curve
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Example dose-response function (logistic function)
def dose_response(x, bottom, top, ic50, hill_slope):
    return bottom + (top - bottom) / (1 + (x / ic50)**hill_slope)



# Remove the 0 dose and corresponding response
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
plt.scatter(doses, responses, label='Data', color='#226E9C')
plt.plot(doses_fit, responses_fit, label='Fit', color='#3C93C2')
plt.axvline(x=ic50, color='#FC4E2A', linestyle='--', label=f'IC50: {ic50:.2f}')
plt.xscale('log')
plt.xlabel('Dose (uM)')
plt.ylabel('Absorbance (Cell Viability)')
plt.legend()
plt.title(plot_title)
plt.savefig('C:/GSC Project/drug_dose_response/figures/' + plot_filename + '.png')
plt.show()









excel_file = 'C:/GSC Project/drug_dose_response/raw_data/temporarySkanitExport May Tue 28 2024 14-36-36-7921036.xlsx'
plot_title = 'G571 (20 min post MTS addition)'

excel_file = 'C:/GSC Project/drug_dose_response/raw_data/temporarySkanitExport May Tue 28 2024 14-36-55-4190108.xlsx'
plot_title = 'G571 (40 min post MTS addition)'

excel_file = 'C:/GSC Project/drug_dose_response/raw_data/temporarySkanitExport May Tue 28 2024 14-37-19-9378918.xlsx'
plot_title = 'G571 (60 min post MTS addition)'


excel_file = 'C:/GSC Project/drug_dose_response/raw_data/temporarySkanitExport May Tue 28 2024 15-33-20-4853163.xlsx'
plot_title = 'G571 (180 min post MTS addition)'
# plot_title = 'HeLa (180 min post MTS addition)'




plot_filename = sanitize_filename(plot_title)
plot_filename = plot_filename.replace(' ', '_')

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(excel_file)

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

# Cell number G571
mean_abs = plate_only_df.iloc[0, :]
# Cell number HeLa
# mean_abs = plate_only_df.iloc[2, :]

# Cell numbers
concentrations = [50000, 25000, 12500, 6250, 3125, 1563, 781, 391, 195, 0]

def linear_response(x, slope, intercept):
    return slope * x + intercept

# Remove the 0 dose and corresponding response
doses = np.array(concentrations[:], dtype=float)
responses = np.array(mean_abs[:], dtype=float)

# Fit the model to the data
doses_filtered = doses[2:]
responses_filtered = responses[2:]

popt, pcov = curve_fit(linear_response, doses_filtered, responses_filtered)

# Extract the fitted parameters
slope, intercept = popt
print(f"Slope: {slope}, Intercept: {intercept}")

# Calculate the fitted responses
fitted_responses = linear_response(doses_filtered, slope, intercept)
fitted_responses = np.array(list(map(lambda x: linear_response(x, *popt), doses_filtered)))

# Calculate the correlation coefficient (R value)
correlation_matrix = np.corrcoef(responses_filtered, fitted_responses)
correlation_xy = correlation_matrix[0, 1]
r_value = correlation_xy
print(f"R value: {r_value}")

# Generate doses for plotting the fit
# Use a range that covers all data points
doses_fit = np.linspace(min(doses_filtered), max(doses_filtered), 100)
responses_fit = linear_response(doses_fit, *popt)

# Plot the data and the fit
plt.figure(figsize=(8, 6))
plt.scatter(doses, responses, label='Data', color='#226E9C')
plt.plot(doses_fit, responses_fit, label='Fit', color='#3C93C2')
plt.xlabel('Cell Number (Cells/well)')
plt.ylabel('Absorbance (Cell Viability)')
plt.legend()
plt.title(plot_title)
plt.annotate(f'R = {r_value:.4f}', xy=(0.95, 0.1), xycoords='axes fraction', fontsize=12, color='black', ha='right', va='bottom')
plt.savefig('C:/GSC Project/drug_dose_response/figures/' + plot_filename + '.png')
plt.show()

