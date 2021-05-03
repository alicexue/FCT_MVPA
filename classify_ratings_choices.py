#!/usr/bin/env python
"""
based on a script from ab
modified by ax for FCT - june 11, 2019

includes methods for performing within-task classification, searchlight analyses, and cross-task classification
scikit-learn and pymvpa are used to perform classification (python 2.7 - pymvpa primarily supported py2.7)
(pymvpa makes it easier to generate null distributions for permutation tests and run searchlight analyses;
scikit-learn has a function that provides classifier evidence/probability of class A vs. B)
"""

import sklearn.svm, numpy, os, nibabel, sys
from sklearn.model_selection import cross_val_score
import random
import statsmodels.api as sm
from mvpa2.suite import *
from mvpa2.datasets.miscfx import coarsen_chunks
import pickle

# custom helper functions
from trial_class_median_split import *

"""
# INPUT ARGUMENTS
# subjid: subject ID
# behav_dir: String pointing to behavioral data directory
# level 1 model dir: String pointing to level 1 model output
# train: imaging data to train classifier on
# target: label to decode from training data
# ROIname: name of ROI (based on ROI filenames)
 
# searchlight is always whole brain at the moment - ROIname input is ignored. 
# In the do_searchlight function, brain data is read in and then a whole brain mask is applied to it.

# cross task: train classifier on "train" input and test classifer on "test" output 
# (where train != test; i.e. decode taste ratings from health activity)
"""

if len(sys.argv) < 7:
	print "usage: classify_ratings_choices.py <subjid> <behav_dir> <level 1 model_dir> <train: taste/health/foodchoice> <target: taste/health/objective/subjective/objective_regress/subjective_regress/visual_features> <ROIname: lOFC/mOFC/OFC> <searchlight/cross_task(optional)>"
	sys.exit(-1)

subjid=sys.argv[1]
behav_dir=sys.argv[2]
path=sys.argv[3]
train_rating = sys.argv[4]
target_rating = sys.argv[5]
ROI = sys.argv[6] # lOFC or mOFC or OFC or V1
do_searchlight = False
do_cross_task_analysis = False
if len(sys.argv) == 8 and sys.argv[7] == 'searchlight':
	do_searchlight = True
elif len(sys.argv) == 8 and sys.argv[7] == 'cross_task':
	do_cross_task_analysis = True

# if we want to predict health or taste (target_rating/behav_rating) from choice data
# first we want to train classifiers to predict behav_rating from behav_rating imaging data
if train_rating == 'foodchoice' or do_cross_task_analysis: 
	imaging_rating = target_rating
else:
	imaging_rating = train_rating

behav_rating = target_rating

do_visual_feature_searchlight = False
visual_features = ['red_intensity','green_intensity','blue_intensity','luminance','contrast','hue','saturation','brightness']
if do_searchlight and behav_rating in visual_features: # do searchlight on one of the visual features
	do_visual_feature_searchlight = True


"""
Run cross-validation procedure
outputdir: String denoting path to desired output directory
X: training data features
y: training data labels
y_name: String denoting what the labels refer to 
X_name: String denoting which task the training data is from
"""
def run_cross_validation(outputdir,X,y,y_name,X_name):

	"""
	# shuffle training AND test labels in cross validation (not desired)
	ds = dataset_wizard(X, targets=y, chunks=range(len(y)))
	clf = LinearCSVMC(C=1)
	coarsen_chunks(ds,nchunks=4)
	random.shuffle(ds.chunks)
	print ds.chunks
	partitioner = NFoldPartitioner() 
	permutator = AttributePermutator('targets', count=1000)
	distr_est = MCNullDist(permutator, tail='right', enable_ca=['dist_samples'])
	cv_mc = CrossValidation(clf,partitioner,postproc=mean_sample(),null_dist=distr_est,enable_ca=['stats'],errorfx=lambda p, t: numpy.mean(p == t))
	acc = cv_mc(ds)
	print "mean accuracy:", numpy.mean(acc)
	print cv_mc.ca.stats.stats['ACC']
	print numpy.ravel(cv_mc.null_dist.ca.dist_samples)
	p = cv_mc.ca.null_prob
	print "pval:",numpy.asscalar(p)
	"""

	if not os.path.exists(outputdir):
		os.mkdir(outputdir)
	ROI_outputdir = os.path.join(outputdir,'cross_val_scores_' + ROI)
	if not os.path.exists(ROI_outputdir):
		os.mkdir(ROI_outputdir)
	filepath = os.path.join(ROI_outputdir,subjid+'_predict_'+y_name+'_from_'+X_name+'_fmrirsp.txt')

	# during cross validation, shuffle training labels but maintain correct labels for test data
	# http://www.pymvpa.org/tutorial_significance.html
	random.seed(a=0)
	ds = dataset_wizard(X, targets=y, chunks=range(len(y)))
	clf = LinearCSVMC(C=1)
	coarsen_chunks(ds,nchunks=4)
	random.shuffle(ds.chunks)
	print ds.chunks
	partitioner = NFoldPartitioner() 
	
	"""
	# does not work
	ds.chunks = np.ones(len(y)) # 1 chunk
	print ds.chunks
	partitioner = NFoldPartitioner(cvtype=4) # 4 fold partitions?
	"""

	repeater = Repeater(count=1000)
	permutator = AttributePermutator('targets',
		limit={'partitions': 1},
		count=1)
	null_cv = CrossValidation(
		clf, ChainNode([partitioner, permutator], space=partitioner.get_space()),
		postproc=mean_sample(),
		errorfx=mean_match_accuracy)
	distr_est = MCNullDist(repeater, tail='right',
		measure=null_cv,
		enable_ca=['dist_samples'])
	cv_mc_corr = CrossValidation(clf,
		partitioner,
		postproc=mean_sample(),
		null_dist=distr_est,
		enable_ca=['stats'],
		errorfx=mean_match_accuracy)
	acc = cv_mc_corr(ds)
	print "mean accuracy:", numpy.mean(acc)
	#print cv_mc_corr.ca.stats.stats['ACC']
	null_dist_samples = numpy.ravel(cv_mc_corr.null_dist.ca.dist_samples)
	p = cv_mc_corr.ca.null_prob
	print "pval:",numpy.asscalar(p)

	# save cross validation score and the percentage of trials in class 1
	results = [numpy.mean(acc),numpy.mean(y==numpy.ones(len(y)))*100]
	numpy.savetxt(filepath,results,delimiter='\n')
	save_permutation_results(outputdir, null_dist_samples, y_name, X_name)

	#sen = clf.get_sensitivity_analyzer()
	#print sen 
	#weights = sen.raw_results
	#print weights

"""
Run searchlight analysis
outputdir: String denoting path to desired output directory
X: training data features
y: training data labels
y_name: String denoting what the labels refer to 
X_name: String denoting which task the training data is from
outputname: String to denote prefix of output name

Based on pymvpa tutorial
"""
def run_searchlight(outputdir,X,y,y_name,X_name,outputname):
	if not os.path.exists(outputdir):
		os.mkdir(outputdir)

	random.seed(a=0)

	# instead of reading in each trial individually, try reading the nifti file directly
	print '\n reading data in directly'
	filename = os.path.join(path,subjid,"task-"+bidsRating+"_run-1",subjid+"_task-"+bidsRating+"_run-1.feat/"+subjid+"_task-"+bidsRating+"_merged.nii.gz")
	## path currently set to analysis_files, need to point to fmriprep
	#full_brain_mask = os.path.join('..','fmriprep',subjid,'func',subjid+"_task-"+bidsRating+"_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
	full_brain_mask = 'FCT_meanT1_trans.nii.gz'
	print full_brain_mask, full_brain_mask
	if not os.path.exists(full_brain_mask):
		print "Could not find brain mask!"
		sys.exit()

	ds = fmri_dataset(filename, targets=y, chunks=range(len(y)), mask=full_brain_mask)
	print ds.a.imghdr
	print len(ds)
	print ds.nfeatures
	print ds.shape
	print ds.chunks
	coarsen_chunks(ds,nchunks=4) # 4-fold cross validation
	random.shuffle(ds.chunks)
	print ds.chunks

	partitioner = NFoldPartitioner() 
	zscore(ds, dtype='float32')
	clf = LinearCSVMC(C=1)
	# setup measure to be computed by Searchlight
	# cross-validated mean transfer using an N-fold dataset splitter
	cv = CrossValidation(clf, NFoldPartitioner())
	# get ids of features that have a nonzero value

	print 'starting searchlight'
	for radius in [2]:
		print "Running searchlight with radius: %i ..." % (radius)
		sl = sphere_searchlight(cv, radius=radius,
			postproc=mean_sample()) # add postproc to get mean of accuracy scores for each fold of cross validation

		sl_map = sl(ds)
		sl_map.samples *= -1
		sl_map.samples += 1
		niftiresults = map2nifti(sl_map, imghdr=ds.a.imghdr)

		"""
		try:
			pickle.dump(niftiresults, open(os.path.join(outputdir,outputname+"_radius-"+str(radius)+"_niftiresults.p"), "wb"))
		except:
			print 'could not save output to pickle'
		"""

		try:
			niftiresults.to_filename(os.path.join(outputdir,outputname+"_radius-"+str(radius)+"_niftiresults.nii.gz"))
		except:
			print 'could not save output to nii.gz'

"""
Run cross-task classification for choices -- retrieve classifier evidence of taste/health from choices
outputdir: String denoting path to desired output directory
X: training data features
y: training data labels
y_name: String denoting what the labels refer to 
X_name: String denoting which task the training data is from
maskvox: ROI matrix for ROI extraction from choice data
"""
def run_choice_classification(outputdir,X,y,y_name,X_name,maskvox):
	if not os.path.exists(outputdir):
		os.mkdir(outputdir)
	ROI_outputdir = os.path.join(outputdir,'predict_from_choices_' + ROI)
	if not os.path.exists(ROI_outputdir):
		os.mkdir(ROI_outputdir)

	# on habanero, taste/health and choice model-003 outputs are stored in different model directories (model is the same but they were run separately)
	if path.endswith('model-003'):
		choicepath = path+'-choices'
	elif path.endswith('model-003/'):
		choicepath = path[:-1]+'-choices/'

	# sorry, confusing variable name here -- "rating" is for choice rating task
	rating = nibabel.load(os.path.join(choicepath,subjid,"task-foodchoice_run-1",subjid+"_task-foodchoice_run-1.feat/"+subjid+"_task-foodchoice_merged.nii.gz")).get_data()

	choice_trials = get_choice_trials(subjid,behav_dir)
	choice_trials = choice_trials.loc[choice_trials['choice_rating']!=0]
	print choice_trials

	# get choice trials
	ratingm = numpy.zeros((rating.shape[3],maskvox[0].shape[0]))
	for trial in range(len(choice_trials)):
		tmp=rating[:,:,:,trial]
		ratingm[trial,:] = tmp[maskvox]

	#collapse all into ntrials*nvox arrays
	X_choices=ratingm
	#Set nans to 0? SHOULD GET NO NANS
	X_choices[numpy.isnan(X_choices)]=0 # uncommented by alice

	#y_choices = numpy.array(choice_trials['chose_alt'])

	#ds_choices = dataset_wizard(X_choices, targets=y_choices)
	#ds_choices = dataset_wizard(X_choices)

	clf=sklearn.svm.SVC(kernel='linear',probability=True)
	clf.fit(X,y) # train classifier on input (taste/health rating imaging data)
	cv_scores = cross_val_score(clf, X, y, cv=4)
	print "cross validation score for all %s trials:"%(behav_rating), numpy.mean(cv_scores)

	classifier_evidence=clf.predict_proba(X_choices) # retrieve classifier evidence from choice imaging trials
	classifier_predict=clf.predict(X_choices)

	print classifier_evidence
	print classifier_predict

	filepath = os.path.join(ROI_outputdir,subjid+'_predict_'+y_name+'_from_choices_fmrirsp_classifier_evidence.txt')
	numpy.savetxt(filepath,classifier_evidence,delimiter=",")
	filepath = os.path.join(ROI_outputdir,subjid+'_predict_'+y_name+'_from_choices_fmrirsp_classifier_predictions.txt')
	numpy.savetxt(filepath,classifier_predict,delimiter=",")

"""
Run cross-task classification (decode taste from health rating trials; decode health from taste rating trials)
outputdir: String denoting path to desired output directory
X: training data features
y: training data labels
y_name: String denoting what the labels refer to 
X_name: String denoting which task the training data is from
maskvox: ROI matrix for ROI extraction 
"""
def run_cross_task_classification(outputdir,X,y,y_name,X_name,maskvox):
	ROI_outputdir = os.path.join(outputdir,'cross_task_classification_' + ROI)
	if not os.path.exists(outputdir):
		os.mkdir(outputdir)
	if not os.path.exists(ROI_outputdir):
		os.mkdir(ROI_outputdir)

	assert y_name == X_name
	print "Trained classifer on", y_name

	print 'Find evidence of', target_rating, 'from', train_rating

	print "Test", y_name, "classifer on:"
	print os.path.join(path,subjid,"task-"+BIDS_rating_names[train_rating]+"_run-1",subjid+"_task-"+BIDS_rating_names[train_rating]+"_run-1.feat/"+subjid+"_task-"+BIDS_rating_names[train_rating]+"_merged.nii.gz")
	# what we want to test the classifer on - "train_rating" (yes it's very confusing)
	rating = nibabel.load(os.path.join(path,subjid,"task-"+BIDS_rating_names[train_rating]+"_run-1",subjid+"_task-"+BIDS_rating_names[train_rating]+"_run-1.feat/"+subjid+"_task-"+BIDS_rating_names[train_rating]+"_merged.nii.gz")).get_data()

	test_rating_df = get_trial_ratings(subjid,behav_rating_names[train_rating],behav_dir)
	test_rating_df = test_rating_df.loc[test_rating_df[behav_rating_names[train_rating]]!=0]
	print test_rating_df

	# get choice trials
	ratingm = numpy.zeros((rating.shape[3],maskvox[0].shape[0]))
	for trial in range(len(test_rating_df)):
		tmp=rating[:,:,:,trial]
		ratingm[trial,:] = tmp[maskvox]

	#collapse all into ntrials*nvox arrays
	X_test=ratingm
	#Set nans to 0? SHOULD GET NO NANS
	X_test[numpy.isnan(X_test)]=0 # uncommented by alice

	#y_choices = numpy.array(choice_trials['chose_alt'])

	#ds_choices = dataset_wizard(X_choices, targets=y_choices)
	#ds_choices = dataset_wizard(X_choices)

	clf=sklearn.svm.SVC(kernel='linear',probability=True)
	clf.fit(X,y)
	cv_scores = cross_val_score(clf, X, y, cv=4)
	print "cross validation score for all %s trials:"%(behav_rating), numpy.mean(cv_scores)

	classifier_evidence=clf.predict_proba(X_test)
	classifier_predict=clf.predict(X_test)

	print classifier_evidence
	print classifier_predict

	filepath = os.path.join(ROI_outputdir,subjid+'_evidence_of_'+y_name+'_from_'+train_rating+'_fmrirsp_classifier_evidence.txt')
	numpy.savetxt(filepath,classifier_evidence,delimiter=",")
	filepath = os.path.join(ROI_outputdir,subjid+'_evidence_of_'+y_name+'_from_'+train_rating+'_fmrirsp_classifier_predictions.txt')
	numpy.savetxt(filepath,classifier_predict,delimiter=",")

def save_permutation_results(outputdir,null_dist_samples,y_name,X_name):
	if not os.path.exists(outputdir):
		os.mkdir(outputdir)
	outputdir = os.path.join(outputdir,'cross_val_scores_permutation_results_' + ROI)
	print null_dist_samples
	if not os.path.exists(outputdir):
		os.mkdir(outputdir)
	filepath = os.path.join(outputdir,subjid+'_predict_'+y_name+'_from_'+X_name+'_fmrirsp_permutation_results.txt')
	numpy.savetxt(filepath,null_dist_samples,delimiter='\n')

def save_classification_weights(outputdir,classifier,y_name,X_name):
	outputdir = os.path.join(outputdir,'cross_val_scores_classification_weights_' + ROI)
	if not os.path.exists(outputdir):
		os.mkdir(outputdir)
	filepath = os.path.join(outputdir,subjid+'_predict_'+y_name+'_from_'+X_name+'_fmrirsp_classification_weights.txt')
	numpy.savetxt(filepath,classifier.coef_,delimiter='\n')


# nutrient_type: subjective/objective
def run_regress_out_rating_analysis(nutrient_type):
	if nutrient_type == 'objective':
		nutrient_options = ['Kcal','Carbohydrates (g)','Protein (g)','Fat (g)','% Carb. per Kcal','% Protein per Kcal',
							'% Fat per Kcal','Energy Density','Na (mg)','Total Sugar (g)','Gluten']
		outputdir = './cross_val_scores_model-003/objective_nutrient_analysis_regress_out_value'
	elif nutrient_type == 'subjective':
		nutrient_options = ['Calories','Carbohydrates','Fat','Gluten','Protein','Sodium','Sugar','Vitamins']
		outputdir = './cross_val_scores_model-003/subjective_nutrient_analysis_regress_out_value'
	else:
		print "Invalid nutrient_type: objective or subjective"
		sys.exit()
	for nutrient in nutrient_options:
		y = []
		imaging_rating_df = get_trial_ratings(subjid,behav_rating_names[imaging_rating],behav_dir)

		if nutrient_type == "objective":
			nutrient_df = get_nutrient_df()
		elif nutrient_type == "subjective":
			nutrient_df = get_subjective_nutrient_df()
		imaging_rating_df = imaging_rating_df.merge(nutrient_df,left_on='food',right_on='stimulus')

		# run linear regression
		mod=sm.OLS(imaging_rating_df[behav_rating_names[imaging_rating]],imaging_rating_df[nutrient])
		res = mod.fit()
		imaging_rating_df['resid']=res.resid
		print imaging_rating_df

		trial=0
		validimagingtrials = imaging_rating_df.loc[imaging_rating_df[behav_rating_names[imaging_rating]]!=0]

		ratingm = numpy.zeros((len(validimagingtrials),maskvox[0].shape[0]))

		print validimagingtrials
		print len(validimagingtrials)

		median = numpy.median(imaging_rating_df['resid'])

		validimagingtrialresetindex = 0
		for trial in validimagingtrials.index.values:						
			food = imaging_rating_df.loc[trial,'food']
			imaging_trial_rating = imaging_rating_df.loc[trial,behav_rating_names[imaging_rating]]
			# get health/taste rating of item based on what imaging_rating is

			print nutrient
			print "trial:", trial, "validimagingtrialresetindex:", validimagingtrialresetindex, "imaging_trial_rating:", imaging_trial_rating

			food_resid = imaging_rating_df.loc[trial,'resid']
			if food_resid < median:
				trialBinN = 0
			else:
				trialBinN = 1
			y.append(trialBinN)
			print food, trial, validimagingtrialresetindex, trialBinN, '\n'

			#collapse all into ntrials*nvox arrays
			tmp=rating[:,:,:,validimagingtrialresetindex]
			ratingm[validimagingtrialresetindex,:] = tmp[maskvox]
			validimagingtrialresetindex+=1

		#collapse all into ntrials*nvox arrays
		X=ratingm
		#Set nans to 0? SHOULD GET NO NANS
		X[numpy.isnan(X)]=0 # uncommented by alice

		y = numpy.array(y)
		print len(y)
		print "percent of trials in first bin/class:", numpy.mean(y==numpy.zeros(len(y)))*100

		print X.shape

		#Train classifier 
		y_name = nutrient.replace(' ', '-')
		X_name = imaging_rating
		run_cross_validation(outputdir,X,y,y_name,X_name)


##########################################
# Process input and run desired analysis #
##########################################

print 'Predict', behav_rating, 'from', imaging_rating

BIDS_rating_names = {'health':'healthrating','taste':'tasterating','foodchoice':'choicerating'}
behav_rating_names = {'health':'health_rating','taste':'taste_rating','foodchoice':'choice_rating'}

bidsRating = BIDS_rating_names[imaging_rating]

existingROIs = ["lOFC","mOFC","OFC","V1"]

if ROI in existingROIs:
	mask=nibabel.load(os.path.join('ROI','aal_ROIs',ROI+'_MNI2009.nii.gz')).get_data()
	print ROI+"_MNI2009.nii.gz"
else:
	print "ERROR: ROI not defined. Here are possible options:"
	print existingROIs
	sys.exit()

maskvox=numpy.where(mask>0)

print "mask.shape",mask.shape

rating = nibabel.load(os.path.join(path,subjid,"task-"+bidsRating+"_run-1",subjid+"_task-"+bidsRating+"_run-1.feat/"+subjid+"_task-"+bidsRating+"_merged.nii.gz")).get_data()

print "rating.shape",rating.shape
# rating.shape = (a, b, c, ntrials)


if behav_rating in behav_rating_names: # decode taste or health (not nutrients/visual features) OR predict taste/health from choices
	startOfSecondBin = get_second_bin_starting_rating(subjid,behav_rating_names[behav_rating],behav_dir) # get the adjusted median
	print "median:", startOfSecondBin
	y = []

	behav_rating_df = get_trial_ratings(subjid,behav_rating_names[behav_rating],behav_dir)
	#if imaging_rating != behav_rating: # then need to find corresponding behav_ratings for each imaging_rating trial
	imaging_rating_df = get_trial_ratings(subjid,behav_rating_names[imaging_rating],behav_dir)

	print rating.shape[3] # number of trials

	# get imaging_rating and behav_rating for each food and in the order of the imaging trials
	combined_df = imaging_rating_df
	combined_df['trialN'] = combined_df.index.values
	if imaging_rating != behav_rating:
		combined_df = combined_df.merge(behav_rating_df, left_on='food', right_on='food')

	print combined_df 

	# need to check that rating on imaging trial is valid because beta series returns results for valid imaging trials only
	nvalidtrials = len(combined_df[(combined_df[behav_rating_names[imaging_rating]] != 0) & (combined_df[behav_rating_names[behav_rating]] != 0)])

	print "nvalidtrials",nvalidtrials

	ratingm = numpy.zeros((nvalidtrials,maskvox[0].shape[0]))
	trial=0
	validimagingtrials = imaging_rating_df.loc[imaging_rating_df[behav_rating_names[imaging_rating]]!=0]

	print "len(validimagingtrials)",len(validimagingtrials)

	print "\n"

	# get training data
	validtrialindex = 0 # keep track of index of imaging trial where behav_trial_rating != 0, in order to set ratingm
	validimagingtrialresetindex = 0
	for trial in validimagingtrials.index.values:		
		food = combined_df.loc[trial,'food']
		imaging_trial_rating = combined_df.loc[trial,behav_rating_names[imaging_rating]]
		# get health/taste rating of item based on what imaging_rating is

		behav_trial_rating = combined_df.loc[trial,behav_rating_names[behav_rating]]
		# get health/taste rating of item based on what behav_rating is

		if behav_trial_rating != 0:
			if behav_trial_rating < startOfSecondBin:
				trialBinN = 0
			else:
				trialBinN = 1
			y.append(trialBinN)

			print "trial:", trial, "food:", food
			print "imaging_rating", imaging_rating
			print "behav_rating", behav_rating
			print "trialBinN:", trialBinN 
			print "validimagingtrialresetindex:", validimagingtrialresetindex
			print "imaging_trial_rating:", imaging_trial_rating
			print "behav_trial_rating:", behav_trial_rating
			print "validtrialindex:", validtrialindex

			print '\n'

			#collapse all into ntrials*nvox arrays
			tmp=rating[:,:,:,validimagingtrialresetindex]
			ratingm[validtrialindex,:] = tmp[maskvox]
			validtrialindex+=1
		validimagingtrialresetindex+=1

	assert validtrialindex == ratingm.shape[0]
	print validtrialindex, rating.shape[3]
	#collapse all into ntrials*nvox arrays
	print ratingm.shape
	ratingdata = ratingm
	X=ratingdata
	
	#Set nans to 0? SHOULD GET NO NANS
	X[numpy.isnan(X)]=0 # uncommented by alice

	y = numpy.array(y)
	print y
	print len(y)
	print "percent of trials in first bin/class:", numpy.mean(y==numpy.zeros(len(y)))*100

	print X.shape

	#Train classifier 
	if train_rating != 'foodchoice':
		# predict health/taste from health/taste
		y_name = behav_rating
		X_name = imaging_rating
		if not do_searchlight and not do_cross_task_analysis:
			outputdir = './cross_val_scores_model-003'
			run_cross_validation(outputdir,X,y,y_name,X_name)
		elif do_searchlight:
			outputdir = './searchlight_outputs_model-003'
			outputname = subjid+'_predict_'+y_name+'_from_'+X_name+'_searchlight'
			run_searchlight(outputdir,X,y,y_name,X_name,outputname)
		elif do_cross_task_analysis:
			outputdir = './cross_task_classification_model-003'
			run_cross_task_classification(outputdir,X,y,y_name,X_name,maskvox)

	else: # predict health/taste from choices
		y_name = behav_rating
		X_name = imaging_rating
		outputdir = './predictions_from_choices_model-003'
		run_choice_classification(outputdir,X,y,y_name,X_name,maskvox)

# decode nutrients or visual features
elif behav_rating == 'objective' or behav_rating == 'subjective' or (not do_searchlight and behav_rating == 'visual_features'):	
	if behav_rating == 'objective':
		options =  ['Kcal','Carbohydrates (g)','Protein (g)','Fat (g)','% Carb. per Kcal','% Protein per Kcal',
					'% Fat per Kcal','Energy Density','Na (mg)','Total Sugar (g)','Gluten']
	elif behav_rating == 'subjective':
		options =  ['Calories','Carbohydrates','Fat','Gluten','Protein','Sodium','Sugar','Vitamins']
	elif behav_rating == 'visual_features':
		options = ['red_intensity','green_intensity','blue_intensity',
				   'luminance','contrast','hue','saturation','brightness']

	for feature in options:
		y = []
		imaging_rating_df = get_trial_ratings(subjid,behav_rating_names[imaging_rating],behav_dir)
		
		trial=0
		validimagingtrials = imaging_rating_df.loc[imaging_rating_df[behav_rating_names[imaging_rating]]!=0]

		ratingm = numpy.zeros((len(validimagingtrials),maskvox[0].shape[0]))

		print validimagingtrials
		print len(validimagingtrials)

		validimagingtrialresetindex = 0
		for trial in validimagingtrials.index.values:						
			food = imaging_rating_df.loc[trial,'food']
			imaging_trial_rating = imaging_rating_df.loc[trial,behav_rating_names[imaging_rating]]
			# get health/taste rating of item based on what imaging_rating is

			print feature
			print "trial:", trial, "validimagingtrialresetindex:", validimagingtrialresetindex, "imaging_trial_rating:", imaging_trial_rating

			if behav_rating == 'objective':
				if feature == 'Gluten':
					category, median = get_nutrient_info(food,feature) # 1 if no gluten, 2 if yes gluten
					y.append(category)
				else:
					trialBinN = get_nutrient_info_bin(food,feature)
					y.append(trialBinN)	
			elif behav_rating == 'subjective':
				category, median = get_subjective_nutrient_info(food,feature)
				if category < median:
					trialBinN = 0
				else:
					trialBinN = 1
				y.append(trialBinN)					
			elif behav_rating == 'visual_features':
				category, median = get_stimulus_visual_features(food,feature)
				print category
				if category < median:
					trialBinN = 0
				else:
					trialBinN = 1
				y.append(trialBinN)	

			print food, trial, validimagingtrialresetindex, trialBinN, '\n'

			#collapse all into ntrials*nvox arrays
			tmp=rating[:,:,:,validimagingtrialresetindex]
			ratingm[validimagingtrialresetindex,:] = tmp[maskvox]
			validimagingtrialresetindex+=1

		X=ratingm
		#Set nans to 0? SHOULD GET NO NANS
		X[numpy.isnan(X)]=0 # uncommented by alice

		y = numpy.array(y)
		print len(y)
		print "percent of trials in first bin/class:", numpy.mean(y==numpy.ones(len(y)))*100

		print X.shape

		#Train classifier 
		y_name = feature.replace(' ', '-')
		y_name = y_name.replace('_', '-')
		X_name = imaging_rating
		
		outputdir = './cross_val_scores_model-003'
		run_cross_validation(outputdir,X,y,y_name,X_name)

# perform searchlight analysis
elif do_searchlight and do_visual_feature_searchlight: # do searchlight on a visual feature
	feature = behav_rating
	y = []
	imaging_rating_df = get_trial_ratings(subjid,behav_rating_names[imaging_rating],behav_dir)
	
	trial=0
	validimagingtrials = imaging_rating_df.loc[imaging_rating_df[behav_rating_names[imaging_rating]]!=0]

	ratingm = numpy.zeros((len(validimagingtrials),maskvox[0].shape[0]))

	print validimagingtrials
	print len(validimagingtrials)

	validimagingtrialresetindex = 0
	for trial in validimagingtrials.index.values:						
		food = imaging_rating_df.loc[trial,'food']
		imaging_trial_rating = imaging_rating_df.loc[trial,behav_rating_names[imaging_rating]]

		print feature
		print "trial:", trial, "validimagingtrialresetindex:", validimagingtrialresetindex, "imaging_trial_rating:", imaging_trial_rating

		category, median = get_stimulus_visual_features(food,feature)
		print feature+":", category
		if category < median:
			trialBinN = 0
		else:
			trialBinN = 1
		y.append(trialBinN)	

		print food, trial, validimagingtrialresetindex, trialBinN, '\n'

		#collapse all into ntrials*nvox arrays
		tmp=rating[:,:,:,validimagingtrialresetindex]
		ratingm[validimagingtrialresetindex,:] = tmp[maskvox]
		validimagingtrialresetindex+=1

	X=ratingm
	#Set nans to 0? SHOULD GET NO NANS
	X[numpy.isnan(X)]=0 # uncommented by alice

	y = numpy.array(y)
	print len(y)
	print "percent of trials in first bin/class:", numpy.mean(y==numpy.ones(len(y)))*100

	print X.shape
	print y

	#Train classifier 
	outputdir = './visual_feature_searchlight_model-003'
	y_name = feature.replace(' ', '-')
	y_name = y_name.replace('_', '-')
	X_name = imaging_rating
	outputname = subjid+'_predict_'+y_name+'_from_'+X_name+'_searchlight'
	run_searchlight(outputdir,X,y,y_name,X_name,outputname)

# decode nutrients and regress out the effect of taste/health ratings for classification of objective nutrients
elif behav_rating == "objective_regress":
	run_regress_out_rating_analysis('objective')

# decode nutrients and regress out the effect of taste/health ratings for classification of subjective nutrients
elif behav_rating == "subjective_regress":
	run_regress_out_rating_analysis('subjective')

