from BioHCI.data.SubjectSpecificData import SubjectSpecificData

# This class uses as a label the extraversion score achieved by each subject in the Big 5 Personality Test
class Boredom_ExtraversionData(SubjectSpecificData):

	# information obtained from file BOYER_ALL_DATA.xlsx
	def create_categories(self):
		print("\nExtraversion Data Processing (subclass of SubjectSpecificData) object is being initialized...")
		print("Categories are not unique in this type of problem - they are scores achieved " +
                        "by each subject. Regression rather than classification should be used.")

		self._categories = [26,22,33,27,47,32,27,28,33,41,40,33,36,36,32,29,34,25,31,
					33,30,33,43,31,34,23,37,24,28,35]

	def get_dataset_name(self):
		return "Extraversion Data"


