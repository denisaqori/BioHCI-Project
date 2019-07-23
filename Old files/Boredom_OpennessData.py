from BioHCI.data.SubjectSpecificData import SubjectSpecificData


# This class uses as a label the openness score achieved by each subject in the Big 5 Personality Test
class Boredom_OpennessData(SubjectSpecificData):

	# information obtained from file BOYER_ALL_DATA.xlsx
	def create_categories(self):
		print("\nOpenness Data Processing (subclass of SubjectSpecificData Processing) object is being initialized...")
		print("In this particular case, we do not record categories, but scores achieved " +
			  "by each subject. This is a regression rather than a classiyfication problem. ")

		self._categories = [30, 29, 29, 30, 27, 31, 23, 31, 30, 28, 28, 23, 31, 32, 32, 24, 27, 32, 32,
						   20, 24, 23, 22, 32, 30, 30, 23, 26, 24, 27]

	def get_dataset_name(self):
		return "Openness Data"