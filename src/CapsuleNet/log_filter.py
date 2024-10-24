import re
import pandas as pd
from openpyxl import load_workbook

TEXT_LOGS = r"D:\Programming Files\Python Files\Deep learning - traffic signs\24.10.05_Results.txt"
EXCEL_LOGS = r"D:\Programming Files\Python Files\Deep learning - traffic signs\24.10.05_Results.xlsx"
EXCEL_ACC_LOGS = r"D:\Programming Files\Python Files\Deep learning - traffic signs\24.10.05_Accuracy_Results.xlsx"
def filter_loss_log():
    # Initialize lists to store train and test data
    train_data = []
    test_data = []
    # Regular expressions to match train and test logs
    train_pattern = r'Epoch: \[(\d+)\]\[(\d+)/\d+\]\s+.*Loss (\d+\.\d+) \((\d+\.\d+)\)'
    test_pattern  = r'Test: \[(\d+)/\d+\]\s+.*Loss (\d+\.\d+) \((\d+\.\d+)\)'

    # Open and read the log file line by line
    with open(TEXT_LOGS, 'r') as file:
        current_epoch = 0
        for line in file:
            # Check if line matches train log pattern
            train_match = re.search(train_pattern, line)
            if train_match:
                epoch = int(train_match.group(1))
                batch = int(train_match.group(2))
                loss = float(train_match.group(3))
                avg_loss = float(train_match.group(4))
                current_epoch = epoch
                train_data.append({"Epoch": epoch, "Batch": batch, "Loss": loss, "Avg Loss": avg_loss})

            # Check if line matches test log pattern
            test_match = re.search(test_pattern, line)
            if test_match:
                batch = int(test_match.group(1))
                loss = float(test_match.group(2))
                avg_loss = float(test_match.group(3))
                test_data.append({"Epoch": current_epoch, "Batch": batch, "Loss": loss, "Avg Loss": avg_loss})

    # Convert to pandas DataFrame
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Write to Excel file
    with pd.ExcelWriter(EXCEL_LOGS) as writer:
        train_df.to_excel(writer, sheet_name='Train', index=False)
        test_df.to_excel(writer, sheet_name='Test', index=False)

    print("Data saved to Excel file successfully.")

def filter_accuracy_log():
    # Initialize lists to store train and test data
    train_data = []
    test_data = []
    # Regular expressions to match train and test logs
    train_pattern = r'Epoch:\s+\[(\d+)\]\[(\d+)/\d+\].*?Prec@1\s+([\d.]+)\s+\(([\d.]+)\)'
    test_pattern  = r'\*\s+Prec@1\s+([\d.]+)'

    # Open and read the log file line by line
    with open(TEXT_LOGS, 'r') as file:
        current_epoch = 0
        for line in file:
            # Check if line matches train log pattern
            train_match = re.search(train_pattern, line)
            if train_match:
                epoch = int(train_match.group(1))
                batch = int(train_match.group(2))
                loss = float(train_match.group(3))
                avg_loss = float(train_match.group(4))
                current_epoch = epoch
                train_data.append({"Epoch": epoch, "Batch": batch, "Loss": loss, "Avg Loss": avg_loss})

            # Check if line matches test log pattern
            test_match = re.search(test_pattern, line)
            if test_match:
                avg_acc = float(test_match.group(1))
                test_data.append({"Epoch": current_epoch, "Avg Loss": avg_acc})

    # Convert to pandas DataFrame
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Write to Excel file
    with pd.ExcelWriter(EXCEL_ACC_LOGS) as writer:
        train_df.to_excel(writer, sheet_name='Train', index=False)
        test_df.to_excel(writer, sheet_name='Test', index=False)

    print("Data saved to Excel file successfully.")


def calculate_avg(file_path:str, sheet_name: str):
    book = load_workbook(file_path)
    df =  pd.read_excel(file_path, sheet_name)
    last_epoch = None
    avg_data = []
    accumulated_value = 0
    count = 0

    for idx, row in df.iterrows():
        current_epoch = row["Epoch"]
        batch_avg_loss = row["Avg Loss"]

        if current_epoch == last_epoch:
            accumulated_value += batch_avg_loss
            count += 1
        else:
            if last_epoch != None:
                tmp_avg = accumulated_value / count
                avg_data.append({"Epoch": last_epoch, "Avg Loss": tmp_avg})
            
            count = 1
            accumulated_value = batch_avg_loss
            last_epoch = current_epoch
    # Last epoch
    if last_epoch != None:
        avg_epoch_loss = accumulated_value / count
        avg_data.append({"Epoch": last_epoch, "Avg Loss": tmp_avg})
    avg_df = pd.DataFrame(avg_data)
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        writer.book = book
        dest_sheet = "{sheet}_AvgLoss".format(sheet = sheet_name)
        avg_df.to_excel(writer, sheet_name=dest_sheet, index=False)

if __name__ == "__main__":
    calculate_avg(EXCEL_ACC_LOGS, "Test")
    # filter_log()
    # filter_accuracy_log()