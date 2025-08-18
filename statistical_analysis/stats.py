import os



def get_stats_table(df, output_path, filename='stats_table.txt'):
    stats = df.describe().round(2)
    stats_str = stats.to_string()

    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, filename)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(stats_str)
    
    return stats_str

