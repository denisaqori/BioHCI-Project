from BioHCI.data.SubjectSpecificData import SubjectSpecificData

class EEG_UserIDData(SubjectSpecificData):

	def create_categories(self):
		print("\nEEG User Identification Data Processing (subclass of SubjectSpecificData Processing) object is being "
			  "initialized...")
		print("We classify the users individually, with the label being the subject's identity.")

		for subj_number in self._subject_list:
			self.categories.append(subj_number)

	def get_dataset_name(self):
		return "EEG_User_ID_Data"
