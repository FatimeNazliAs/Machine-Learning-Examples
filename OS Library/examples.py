import os

# Get the current working directory (CWD)
cwd = os.getcwd()
print("current directory: ", cwd)




def current_path():
    print("current directory before: ")
    print(os.getcwd())
    print()


current_path()
# Changing the CWD
# os.chdir('../')
current_path()



directory = "nazliii"
parent_dir = "C:/Users/Naz/Desktop/Machine Learning"
path = os.path.join(parent_dir, directory)
os.mkdir(path)
print("directory %s created" % directory)



# get the list of all files and directories in the specified directory.
path = "C:/Users/Naz/Desktop/Machine Learning"
dir_list = os.listdir(path)
print("files and directories in ", path)
print(dir_list)