import sqlite3
import pandas as pd
import argparse
import sys
import os

def convert_sqlite_to_excel(db_path, excel_path=None):
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not excel_path:
        base_name = os.path.splitext(db_path)[0]
        excel_path = f"{base_name}.xlsx"

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        
        # Get all table names
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print(f"No tables found in database: {db_path}")
            return
            
        print(f"Found {len(tables)} table(s). Exporting to '{excel_path}'...")
        
        # Create an Excel writer using openpyxl engine
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for table_name in tables:
                table_name = table_name[0]
                print(f"  Exporting table: {table_name}...")
                
                # Read the table into a pandas DataFrame
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                
                # Excel has a 31 character limit for sheet names
                sheet_name = table_name[:31]
                
                # Write the DataFrame to the excel file module
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
        print(f"\nSuccessfully exported to '{excel_path}'!")
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}", file=sys.stderr)
        sys.exit(1)
    except ModuleNotFoundError as e:
        print(f"Module error: {e}", file=sys.stderr)
        print("Please run: pip install pandas openpyxl", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an SQLite database to an Excel file with each table as a sheet.")
    parser.add_argument("db_path", help="Path to the input SQLite database file (e.g., database.db)")
    parser.add_argument("excel_path", nargs="?", help="Optional path to the output Excel file. If not provided, it will use the input db's name.")
    
    args = parser.parse_args()
    convert_sqlite_to_excel(args.db_path, args.excel_path)
