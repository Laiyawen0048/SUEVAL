import pandas as pd
import os

# ====== 1. Read data ======
file_path = r"C:\Users\沐阳\PycharmProjects\pythonProject3\SUEVAL\Input\Filter\City_Data_Standardized_Results.xlsx"
df = pd.read_excel(file_path)

# ====== 2. Desired column order (and keep only these columns) ======
desired_order = [
    "Year", "City", "Province", "Region",
    "GDP", "Local_exp", "Post_rev", "Wastewater",
    "PIAV", "SIAV", "TIAV", "GIOV", "Domestic_GIOV", "HMT_GIOV", "Foreign_GIOV",
    "Ind_ent", "VAT", "Ind_profit", "Ind_curr_assets",
    "Reg_pop", "Nat_growth_rate", "Pop_density",
    "Nonprivate_emp", "Private_emp", "Unemployed", "Avg_staff",
    "PI_emp", "SI_emp", "TI_emp", "PI_emp_share", "SI_emp_share", "TI_emp_share",
    "Agri_emp", "Mining_emp", "Manu_emp", "Utility_emp", "Const_emp",
    "Transpost_emp", "ICT_emp", "Trade_emp", "Hotel_emp", "Finance_emp",
    "RE_emp", "Lease_emp", "R&D_emp", "Water_emp", "Reservice_emp",
    "Edu_emp", "Health_emp", "Culture_emp", "Public_emp",
    "Total_wages", "Avg_wage",
    "Retail_sales", "Wholesale_sales", "Wholesale_ent", "Savings",
    "FAI", "RE_inv", "Res_inv", "New_projects",
    "Domestic_ent", "HMT_ent", "Foreign_ent", "FDI",
    "Grain_output", "Oilcrop_output", "Veg_output", "Fruit_output",
    "Meat_output", "Dairy_output", "Aquatic_output",
    "Local_rev", "Sci_exp", "Edu_exp",
    "Loans", "Deposits",
    "Univ_num", "Mid_sch_num", "Pri_sch_num",
    "Univ_teachers", "Mid_teachers", "Pri_teachers",
    "Univ_enroll", "Mid_enroll", "Pri_enroll", "Voc_enroll",
    "Libraries", "Theaters", "Library_col", "Books_per_100",
    "Health_inst", "Hospitals", "Hospital_beds", "Physicians",
    "Passenger", "Rail_passenger", "Road_passenger",
    "Freight", "Rail_freight", "Road_freight",
    "Post_offices", "Telecom_rev", "Landline_users", "Mobile_users", "Internet_users",
    "SO2_emission", "Waste_treat_rate", "Sewage_plant_rate",
    "Pension_ins", "Medical_ins", "Unemp_ins",
    "Land_area"
]

# ====== 3. Keep only desired columns and reorder ======
# Columns that exist both in desired_order and in the file
final_cols = [c for c in desired_order if c in df.columns]

# Optional: show missing columns
missing_cols = [c for c in desired_order if c not in df.columns]
if missing_cols:
    print("These columns are not in the file and will be ignored:")
    print(missing_cols)

# Reorder and drop all other columns
df_reordered = df[final_cols].copy()

# ====== 4. Save ======
# Overwrite original file
df_reordered.to_excel(file_path, index=False)

# Or save as a new file
out_path = r"C:\Users\沐阳\PycharmProjects\pythonProject3\SUEVAL\Input\Index_system\City_Data_Standardized_Results_Reordered.xlsx"
df_reordered.to_excel(out_path, index=False)

print("Column order adjusted and only desired features are kept.")