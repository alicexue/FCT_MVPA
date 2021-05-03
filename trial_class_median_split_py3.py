import numpy as np
import pandas as pd
import sys
import os
from scipy.io import loadmat

"""
alice xue
june 11, 2019

two way classfication - median split of ratings
"""
conditions = {'health_rating':'_h_','taste_rating':'_y_','choice_rating':'_c_'}

"""
get number of trials in each bin
binN: array of rating numbers in bin
n_rating_trials: keys are rating values, value is number of trials with that rating
"""
def get_ntrials_per_bin(binN,n_rating_trials):
	s = 0
	for i in binN:
		if i in n_rating_trials.index.values:
			s+=n_rating_trials[i]
	return s

"""
perform (adjusted) median split of rating
ratingType: taste_rating or health_rating

"adjusted" referring to whether the median should be included in first or second bin 
depending on which minimizes the difference in the number of trials in each bin

return the first rating in the second bin 
(in subsequent analyses, can just sort rating data into bins by comparing >= this adjusted median)
"""
def get_second_bin_starting_rating(subjid,ratingType,behav_dir):
	subdir = os.path.join(behav_dir,subjid)
	sub_behav_files = os.listdir(subdir)
	for sub_behav_file in sub_behav_files:
		if (sub_behav_file.endswith('.mat') and conditions[ratingType] in sub_behav_file):
			sub_data = loadmat(os.path.join(behav_dir,subjid,sub_behav_file))
			sub_df = pd.DataFrame()
			sub_df[ratingType] = np.concatenate(sub_data[ratingType], axis=0)
			sub_df['food'] = np.concatenate(sub_data['food'], axis=0)
			sub_df['food'] = np.concatenate(sub_df['food'], axis=0)

			sub_df = sub_df.loc[sub_df[ratingType]!=0]

			ratings = sub_df[ratingType]
			median = int(np.median(ratings))
			
			n_rating_trials = sub_df.groupby(ratingType).count()['food']
			 
			binA1 = range(1,median) # not include median in bin A1
			binA2 = range(median,6) # include median in bin A2
			
			binB1 = range(1,median+1) # include median in bin B1
			binB2 = range(median+1,6) # not include median in bin B1
			
			nBinA1 = get_ntrials_per_bin(binA1,n_rating_trials)
			nBinA2 = get_ntrials_per_bin(binA2,n_rating_trials)
			
			nBinB1 = get_ntrials_per_bin(binB1,n_rating_trials)
			nBinB2 = get_ntrials_per_bin(binB2,n_rating_trials)
			
			startOfSecondBin = median
			if abs(nBinA1 - nBinA2) > abs(nBinB1 - nBinB2):
				startOfSecondBin = median+1
			return startOfSecondBin

"""
return bin number of bin that trial falls in 
"""
def get_trial_class(trialRating,startOfSecondBin):
	if trialRating < startOfSecondBin:
		return 0
	else:
		return 1

def get_trial_ratings(subjid,ratingType,behav_dir):
	subdir = os.path.join(behav_dir,subjid)
	sub_behav_files = os.listdir(subdir)
	for sub_behav_file in sub_behav_files:
		if (sub_behav_file.endswith('.mat') and conditions[ratingType] in sub_behav_file):
			sub_data = loadmat(os.path.join(behav_dir,subjid,sub_behav_file))
			df = pd.DataFrame()
			df['food'] = np.concatenate(sub_data['food'], axis=0)
			df['food'] = np.concatenate(df['food'], axis=0)
			df['fat'] = np.concatenate(sub_data['fat'], axis=0)
			df[ratingType] = np.concatenate(sub_data[ratingType], axis=0)
			if ratingType == 'choice_rating':
				df['ref_food'] = sub_data['ref_food'][0]
			#df = df[df[ratingType] != 0]
			return df

def get_choice_trials(subjid,behav_dir):
	subdir = os.path.join(behav_dir,subjid)
	sub_behav_files = os.listdir(subdir)
	for sub_behav_file in sub_behav_files:
		if (sub_behav_file.endswith('.mat') and conditions['choice_rating'] in sub_behav_file):
			sub_data = loadmat(os.path.join(behav_dir,subjid,sub_behav_file))
			df = pd.DataFrame()
			df['food'] = np.concatenate(sub_data['food'], axis=0)
			df['food'] = np.concatenate(df['food'], axis=0)
			df['choice_rating'] = np.concatenate(sub_data['choice_rating'], axis=0)
			#df['health_rating'] = np.concatenate(sub_data['health_rating'], axis=0)
			#df['taste_rating'] = np.concatenate(sub_data['taste_rating'], axis=0)
			#df = df[df[ratingType] != 0]
			return df

"""
return bin number of bin that trial food falls in (1 or 2)
food: name of stimulus 
nutrient: Kcal,	Carbohydrates, Protein, Fat, % Carb. per Kcal, % Protein per Kcal, % Fat per Kcal, Energy Density
"""
def get_nutrient_info_bin(food,nutrient):
	# median in upper bin
	nutrient_content, median = get_nutrient_info(food,nutrient)
	if nutrient_content < median:
		return 0
	else:
		return 1

"""
return nutrition information for food and median value
"""
def get_nutrient_info(food,nutrient):
	nutrient_info_file = 'FoodChoiceTaskItems_nutrient_analysis.csv'
	if not os.path.exists(nutrient_info_file):
		print("ERROR: Objective nutrient information missing: can't find FoodChoiceTaskItems_nutrient_analysis.csv")
		sys.exit()
	df = pd.read_csv(nutrient_info_file)
	if nutrient not in df.columns:
		print("ERROR: Could not find %s in %s"%(nutrient, nutrient_info_file))
		sys.exit()
	if food not in df['stimulus'].values:
		print("ERROR: Could not find %s in %s"%(food, nutrient_info_file))
		sys.exit()
	df.set_index('stimulus',inplace=True)
	median = np.median(df[nutrient])
	return df.loc[food,nutrient], median


def get_nutrient_df():
	nutrient_info_file = 'FoodChoiceTaskItems_nutrient_analysis.csv'
	if not os.path.exists(nutrient_info_file):
		print("ERROR: Objective nutrient information missing: can't find FoodChoiceTaskItems_nutrient_analysis.csv")
		sys.exit()
	df = pd.read_csv(nutrient_info_file)
	return df

def get_subjective_nutrient_df():
	nutrient_info_file = 'FoodChoiceTaskItems_subjective_nutrient_analysis.csv'
	if not os.path.exists(nutrient_info_file):
		print("ERROR: Objective nutrient information missing: can't find FoodChoiceTaskItems_subjective_nutrient_analysis.csv")
		sys.exit()
	df = pd.read_csv(nutrient_info_file)
	return df

"""
return nutrition information for food and median value
"""
def get_subjective_nutrient_info(food,nutrient):
	nutrient_info_file = 'FoodChoiceTaskItems_subjective_nutrient_analysis.csv'
	if not os.path.exists(nutrient_info_file):
		print("ERROR: Objective nutrient information missing: can't find FoodChoiceTaskItems_subjective_nutrient_analysis.csv")
		sys.exit()
	df = pd.read_csv(nutrient_info_file)
	if nutrient not in df.columns:
		print("ERROR: Could not find %s in %s"%(nutrient, nutrient_info_file))
		sys.exit()
	if food not in df['stimulus'].values:
		print("ERROR: Could not find %s in %s"%(food, nutrient_info_file))
		sys.exit()
	df.set_index('stimulus',inplace=True)
	median = np.median(df[nutrient])
	return df.loc[food,nutrient], median

"""
return visual feature information for food and median value
"""
def get_stimulus_visual_features(food,feature):
	prop_file = 'FCT_stim_properties.csv'
	if not os.path.exists(prop_file):
		print("ERROR: Stimulus visual information missing: can't find FCT_stim_properties.csv")
		sys.exit()
	df = pd.read_csv(prop_file)
	if feature not in df.columns:
		print("ERROR: Could not find %s in %s"%(feature, prop_file))
		sys.exit()
	if food not in df['stimulus'].values:
		print("ERROR: Could not find %s in %s"%(food, prop_file))
		sys.exit()
	df.set_index('stimulus',inplace=True)
	median = np.median(df[feature])
	return df.loc[food,feature], median

if __name__ == "__main__":
	if len(sys.argv) < 4 or sys.argv[3] not in conditions.keys():
		print("usage: python trial_class_median_split.py <behav_dir> <subjid> <ratingType: health_rating/taste_rating/choice_rating>")
		sys.exit()

	behav_dir = sys.argv[1]
	subjid = sys.argv[2]
	ratingType = sys.argv[3]
	print(get_second_bin_starting_rating(subjid,ratingType,behav_dir))