from BioHCI.data.SubjectSpecificData import SubjectSpecificData


# This class uses as a label the agreebleness score achieved by each subject in the Big 5 Personality Test
class Boredom_AgreeblenessData(SubjectSpecificData):

	# information obtained from file BOYER_ALL_DATA.xlsx
	def create_categories(self):
		print("\nAgreebleness Data Processing (subclass of SubjectSpecificData) object is being initialized...")
		print("Categories are not unique in this type of problem - they are scores achieved " +
			  "by each subject. Regression rather than classification should be used.")

		self._categories = [33, 33, 39, 37, 34, 36, 40, 32, 30, 33, 27, 31, 36, 38, 33, 30, 23, 35, 36,
						   26, 28, 31, 30, 28, 39, 29, 29, 33, 32, 26]

	def get_dataset_name(self):
		return "Agreebleness Data"
