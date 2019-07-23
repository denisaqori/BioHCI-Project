from BioHCI.data.SubjectSpecificData import SubjectSpecificData


# This class uses as a label the neuroticism score achieved by each subject in the Big 5 Personality Test
class Boredom_NeuroticismData(SubjectSpecificData):

	# information obtained from file BOYER_ALL_DATA.xlsx
	def create_categories(self):
		print("\nNeuroticism Data Processing (subclass of SubjectSpecificData) object is being initialized...")
		print("Categories are not unique in this type of problem - they are scores achieved " +
			  "by each subject. Regression rather than classification should be used.")

		self._categories = [11, 34, 27, 30, 12, 33, 31, 20, 26, 9, 22, 16, 38, 8, 27, 25, 34, 17, 10,
						   21, 28, 13, 25, 20, 14, 13, 18, 19, 15, 14]

	def get_dataset_name(self):
		return "Neuroticism Data"