# This file contains utilitary fonctions

import os
import shutil
import constants
from datetime import datetime

#BACKUP_DIR = constants.WORK_DIR + '\\backup\\'
BACKUP_DIR = os.path.join(constants.WORK_DIR, 'backup')

def write_file(fileName, *lists):
    """
    Write values in a file.

    Parameters:
        fileName (path): path of the file where values are saved.
        *lists (array): values that are saved, can be several array.

    Returns:
        None
    """
    if constants.DO_BACKUP:
        backup_file(fileName)
    with open(fileName, 'w') as file:
        for elements in zip(*lists):
            ligne = '\t'.join(str(e) for e in elements)
            file.write(ligne + '\n')

def backup_file(fileName):
    """
    Create backup files.

    Parameters:
        fileName (path): path of the file where values are saved.

    Returns:
    None    
    """
    # Create backup folder if it does not exist
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Save the file if it already exists
    if os.path.exists(fileName):
        timestamp_date = datetime.now().strftime("%d_%m_%Y")
        timestamp_minute = datetime.now().strftime("%Hh_%M")
        name = os.path.basename(os.path.splitext(fileName)[0])
        extension = os.path.splitext(fileName)[1]
        backup_subfolder = BACKUP_DIR + '\\' + timestamp_date + '\\'
        #print(f'backup_subfolder = {backup_subfolder}')
        os.makedirs(backup_subfolder, exist_ok=True)
        backup_name = f"{name}_{timestamp_minute}{extension}"
        #print(f'backup_name = {backup_name}')
        backup_path = os.path.join(backup_subfolder, backup_name)
        #print(f'backup_path = {backup_path}')
        shutil.move(fileName, backup_path)
        print(f"Existing file have been saved in: {backup_path}")
