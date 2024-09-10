import pandas as pd
import warnings
import os


warnings.filterwarnings('ignore')

'''Import all the data streams'''
# Source directory where the raw data is stored and the processed data will be saved to+
root_dir = r"C:\UofL - MSI\DARPA\preprocessed"
participant = 'P01/'  # change to "P#""
trial = 'Mugcake_with error/'
#directory = participant+trial
directory = os.path.join(root_dir, participant,trial)


blinks = pd.read_csv(directory+'Blink_time.csv')  # whether a blink is detected (onset) or not (offset) and confidence
right_pupil = pd.read_csv(directory+'RightPup_time.csv')  # diameter of right pupil and confidence
left_pupil = pd.read_csv(directory+'LeftPup_time.csv')  # diameter of left pupil and confidence
bvp = pd.read_csv(directory+'BVP_time.csv')  # BVP values
gsr = pd.read_csv(directory+'GSR_time.csv')  # GSR values

# Matching columns and adding identifiers
blinks['id'] = 'blink'
right_pupil['id'] = 'right_pup'
left_pupil['id'] = 'left_pup'
bvp['conf'] = 100
bvp = bvp[['time', 'value', 'conf', 'ratings']]
bvp['id'] = 'bvp'
gsr['conf'] = 100
gsr = gsr[['time', 'value', 'conf', 'ratings']]
gsr['id'] = 'gsr'

df = pd.concat([blinks, right_pupil, left_pupil, bvp, gsr])
df = df.sort_values('time')
df.reset_index(drop=True, inplace=True)
print(df.head(20))
final_df = pd.DataFrame()
for index, row in df.iterrows():
    if index % 1000 == 0:
        print(f'Progress: {index}/{len(df)}')
    # BVP
    bvp_index = df.loc[(df['id'] == 'bvp') & (df.index <= index)].index
    if bvp_index.size == 0:
        bvp_val = None
    else:
        bvp_val = df.loc[bvp_index[-1], 'value']
    # GSR
    gsr_index = df.loc[(df['id'] == 'gsr') & (df.index <= index)].index
    if gsr_index.size == 0:
        gsr_val = None
    else:
        gsr_val = df.loc[gsr_index[-1], 'value']
    # Blink
    blink_index = df.loc[(df['id'] == 'blink') & (df.index <= index)].index
    if blink_index.size != 0 and blink_index[-1] == index:
        blink_val = 1
    else:
        blink_val = 0
    # Left Pupil
    left_index = df.loc[(df['id'] == 'left_pup') & (df.index <= index)].index
    if left_index.size == 0:
        left_val = None
    else:
        left_val = df.loc[left_index[-1], 'value']
    # Left Pupil
    right_index = df.loc[(df['id'] == 'right_pup') & (df.index <= index)].index
    if right_index.size == 0:
        right_val = None
    else:
        right_val = df.loc[right_index[-1], 'value']
    # Create new row and append
    new_row = {
        'time': row['time'],
        'conf': row['conf'],
        'rating': row['ratings'], 
        'id': row['id'],
        'bvp': bvp_val,
        'gsr': gsr_val,
        'blink': blink_val,
        'left_pupil': left_val,
        'right_pupil': right_val
        }
    #final_df = final_df.append(new_row, ignore_index=True)
    #final_df = pd.concat([final_df, new_row], ignore_index=True)
    #print(new_row)
    if len(final_df) == 0:
        final_df = pd.DataFrame(new_row, index=[0])
    else:
        final_df.loc[len(final_df)] = new_row
    #print(len(final_df))
    if index  == 100:
        break

final_df.to_csv(os.path.join(root_dir, participant,trial, 'full_data.csv'), index=False)
