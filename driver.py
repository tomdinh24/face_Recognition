from faceCam import faceCam
from faceDetect import faceDetect
from faceComp import faceComp

# prompt 
def prompt():
    print("Please select an option below:")
    print("1. face detect")
    print("2. face compare")
    print("3. face cam")
    option = input("Which option would you like to select? ")
    return option

# user want to continue?
def user_exit():
    exit = input("Do you want to continue? (Y/N) \n")
            
    if exit == "Y" or exit == "y":
        user_choice = prompt()
        match_option(user_choice)
            
    elif exit == "N" or exit == "n":
        return False

    else:
        print("Option is not available!")
        user_exit()


# match user input to available options
def match_option(option):
    match option:
        case "1":
            image_path = input("Please enter the image path: ")
            faceDetect(image_path)
        
        case "2":
            image_path1 = input("Please enter the image path #1: ")
            image_path2 = input("Please enter the image path #2: ")
            faceComp(image_path1,image_path2)
        
        case "3":
            faceCam()

        case _ :
            print("Option is not available!")
            user_exit()
        


# main

exit = True

while exit:
    user_choice = prompt()

    match_option(user_choice)

    exit = user_exit()

print("Thank you for your time!!")



