import numpy as np
import random

def get_user_guess(num:int)-> tuple[str, int]:

    try:
        user_guess:int = int(input("Guess a number in the range: "))
    except:
        raise TypeError('Value must be integer') 
    
    guess_status:str = ""

    if user_guess == num:
        guess_status = "correct"
    elif user_guess > num: 
        guess_status = "too high"    
    else:
        guess_status = "too low"

    return  guess_status, user_guess     


# TODO [1]: implement the guessing_game function
def guessing_game(max:int, *, attempts:int)-> tuple[bool, list[int], int]: # hint: return type is tuple[bool, list[int], int]:

    num:int = random.randint(1, max)

    guesses:list[int] = []
    player_won:bool = False
    chosen_int:int = 0

    while(attempts > 0):
        try:
            guess_status, user_guess = get_user_guess(num)
            print(f"Your guess is {guess_status}")
            guesses.append(user_guess)

            attempts -= 1
            print(f"Number of attempts left: {attempts}")

            if(guess_status == "correct"):
                player_won = True
                chosen_int = num
                break
        except:
            print("You must enter an integer") 


    if attempts == 0 and (not player_won):
        print("You are out of attempts :(")
    
    return player_won, guesses, chosen_int      

# TODO [2]: implement the play_game function
def play_game()-> None:
    max_value:int = 20
    attempts:int = 5

    answer = 'yes'
    while(answer == 'yes'):
        print(f"Guess a number in the range 1 - {max_value}")
        player_won, guesses, chosen_int = guessing_game(max_value, attempts=attempts)
        if player_won:
            assert chosen_int in guesses, "Error: Player won but the number is not in the list"
            print("Congrats! You Won!")
            break
        else:
            assert chosen_int not in guesses, "Error: Player lost but the number is in the list"
            print("You lost!")
            answer = input("Do you want to play again? yes/no: ").lower()

    return        




            

