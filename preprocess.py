import os
import shutil

def process_directories(start_path='.', target_path='.'):
    for root, dirs, files in os.walk(start_path, topdown=True):
        # Handle 'arts' directories
        if 'art' in dirs:
            arts_path = os.path.join(root, 'art')
            try:
                art_files = os.listdir(arts_path)
                if len(art_files) < 10:
                    print(f"Skipping {arts_path} due to insufficient files.")
                    continue                    
                    
                for item in art_files:
                    item_path = os.path.join(arts_path, item)
                    dest_path = os.path.join(target_path, os.path.basename(root), item)
                    if not os.path.exists(os.path.dirname(dest_path)):
                        os.makedirs(os.path.dirname(dest_path))
                    if os.path.exists(dest_path):
                        print(f"Destination already exists: {dest_path}, skipping.")
                        continue
                    
                    os.link(item_path, dest_path)

                print(f"Extracted contents from: {arts_path}")
            except Exception as e:
                print(f"Error processing {arts_path}: {e}")


if __name__ == "__main__":
    process_directories('/home/gjmrem/Yandex.Disk/ /Anime', '/home/gjmrem/datasets/Anime')

    print("Directory processing completed.")