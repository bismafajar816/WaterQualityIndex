import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Input Data
data = {
    "Parameter": ["EC", "pH", "Turbidity", "Hardness", "Nitrate", "Calcium", "Mg", "Cl", "SO4", "K", "TDS", "Fe", "F", "COD", "BOD", "DO", "As", "Cu", "Mn", "Cr", "Pb", "Boron", "Hg", "Total Coliforms", "Fecal Coliforms"],
    "Unit": ["µS/cm", "-", "NTU", "mg/L", "mg/L", "mg/L", "mg/L", "mg/L", "mg/L", "mg/L", "mg/L", "mg/L", "mg/L", "mg/L", "mg/L", "mg/L", "µg/L", "mg/L", "mg/L", "µg/L", "µg/L", "mg/L", "µg/L", "cfu/mL", "cfu/mL"],
    "Permissible_Limit": [None, (6.5, 8.5), 5, 500, 10, None, None, 250, None, None, 1000, 0.3, 1.5, 150, 80, 5, 50, 2, 0.5, 50, 50, 2.5, 1, 0, 0],
    "Wet_Season": [(498, 2700), (7.53, 7.85), (23, 157), (160, 500), (0.9, 3.1), (40, 120), (15, 49), (38, 720), (96, 130), (4.6, 160), (274, 1485), (0, 2), (0.38, 1.2), (80, 250), (23, 70), (6, 7), (2, 3), (0, 0), (0, 6.2), (0, 0), (0, 0), (0.071, 0.4), (0, 0), (7, 84), (2, 25)],
    "Dry_Season": [(462, 4790), (7, 8), (8, 65), (180, 800), (2, 22), (40, 120), (19, 122), (20, 1100), (34, 257), (5, 270), (254, 2635), (0, 0), (0, 2), (60, 210), (15, 80), (5, 7), (3, 7), (0, 0), (0, 0), (0, 0), (0, 0), (0.027, 0.9), (0, 0), (10, 74), (5, 25)]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Define revised weights based on impact
weights = {
    "EC": 2,
    "pH": 4,
    "Turbidity": 3,
    "Hardness": 2,
    "Nitrate": 4,
    "Calcium": 2,
    "Mg": 2,
    "Cl": 3,
    "SO4": 3,
    "K": 2,
    "TDS": 3,
    "Fe": 3,
    "F": 4,
    "COD": 3,
    "BOD": 3,
    "DO": 5,
    "As": 5,
    "Cu": 3,
    "Mn": 3,
    "Cr": 5,
    "Pb": 5,
    "Boron": 3,
    "Hg": 5,
    "Total Coliforms": 4,
    "Fecal Coliforms": 5
}

# Add weights to the dataframe
df["Weight"] = df["Parameter"].map(weights)

# Step 3: Calculate relative weights (Wi)
df["Relative_Weight"] = df["Weight"] / df["Weight"].sum()

# Step 4: Calculate quality ratings (qi) for Wet Season
def calculate_qi(row, season="Wet_Season"):
    permissible_limit = row["Permissible_Limit"]
    value_range = row[season]
    
    if permissible_limit is None or value_range is None:
        return None  # Skip if no limit or values
    
    # Handle cases where permissible_limit is a tuple or zero
    if isinstance(permissible_limit, tuple):
        permissible_limit = sum(permissible_limit) / 2  # Average of range
    
    # If permissible limit is zero or invalid, return None
    if permissible_limit == 0:
        return None
    
    avg_value = sum(value_range) / 2  # Average of min and max
    return (avg_value / permissible_limit) * 100

df["Quality_Rating_Wet"] = df.apply(calculate_qi, season="Wet_Season", axis=1)

# Step 5: Calculate sub-index (SIi)
df["Sub_Index_Wet"] = df["Quality_Rating_Wet"] * df["Relative_Weight"]

# Step 6: Calculate WQI for Wet Season
WQI_wet = df["Sub_Index_Wet"].sum()

# Print Results
print(f"WQI for Wet Season: {WQI_wet}")
print(df[["Parameter", "Quality_Rating_Wet", "Sub_Index_Wet"]])

# Step 7: Visualization (Graphs)
# 7.1 Plot Quality Ratings for Wet Season
plt.figure(figsize=(10, 6))
plt.bar(df["Parameter"], df["Quality_Rating_Wet"], color='skyblue')
plt.title('Water Quality Ratings (Wet Season)')
plt.xlabel('Parameter')
plt.ylabel('Quality Rating (%)')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

# 7.2 Plot Sub-Index for Wet Season
plt.figure(figsize=(10, 6))
plt.bar(df["Parameter"], df["Sub_Index_Wet"], color='lightgreen')
plt.title('Sub-Index for Wet Season')
plt.xlabel('Parameter')
plt.ylabel('Sub-Index')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 7.3 Plot WQI for Wet Season
# The overall WQI score can also be visualized.
plt.figure(figsize=(6, 6))
plt.bar(["Wet Season"], [WQI_wet], color='salmon')
plt.title('Water Quality Index (Wet Season)')
plt.ylabel('WQI')
plt.tight_layout()
plt.show()

# 7.4 Plot Quality Ratings for Dry Season
df["Quality_Rating_Dry"] = df.apply(calculate_qi, season="Dry_Season", axis=1)
plt.figure(figsize=(10, 6))
plt.bar(df["Parameter"], df["Quality_Rating_Dry"], color='lightcoral')
plt.title('Water Quality Ratings (Dry Season)')
plt.xlabel('Parameter')
plt.ylabel('Quality Rating (%)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 7.5 Plot Comparison between Wet and Dry Seasons
df_comparison = df[["Parameter", "Quality_Rating_Wet", "Quality_Rating_Dry"]]
df_comparison.set_index("Parameter", inplace=True)
df_comparison.plot(kind='bar', figsize=(10, 6))
plt.title('Comparison of Quality Ratings: Wet vs Dry Season')
plt.ylabel('Quality Rating (%)')
plt.tight_layout()
plt.show()

# 7.6 Correlation Heatmap
# Correlation matrix for parameters (using the quality ratings for Wet and Dry seasons)
correlation_data = df_comparison.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap between Wet and Dry Season Quality Ratings')
plt.tight_layout()
plt.show()
