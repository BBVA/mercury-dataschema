import os, shutil

from mercury.dataschema.create_tutorials import create_tutorials

def test_create_tutorials():
	# Silently remove the full tree './dataschema_tutorials/'
	shutil.rmtree('./dataschema_tutorials', ignore_errors = True)

	create_tutorials('./')

	assert os.path.isfile('./dataschema_tutorials/hello_dataschema.ipynb')

	# Clean up
	shutil.rmtree('./dataschema_tutorials', ignore_errors = True)
