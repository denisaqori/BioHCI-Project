from BioHCI.data.SubjectSpecificData import SubjectSpecificData

# This class uses as a label the conscientiousness score achieved by each subject in the Big 5 Personality Test
class Boerdom_ConscientiousnessData(SubjectSpecificData):

	# information obtained from file BOYER_ALL_DATA.xlsx
	def create_categories(self):
		print("\nConscientiousness Data Processing (subclass of SubjectSpecificData) object is being initialized...")
		print("Categories are not unique in this type of problem - they are scores achieved " +
			  "by each subject. Regression rather than classification should be used.")

		self._categories = [29, 25, 42, 34, 40, 45, 30, 28, 37, 43, 31, 39, 29, 48, 25, 29, 23, 33, 32,
						   27, 32, 30, 36, 25, 32, 31, 32, 30, 27, 21]

	def get_dataset_name(self):
		return "Conscientiousness Data"
