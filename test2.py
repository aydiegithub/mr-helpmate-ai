from src.backend.intent_moderation_check import (IntentCheck, ModerationCheck)

mod_chk = ModerationCheck()
int_cnf = IntentCheck()

question1 = "I want to kill a person."
question2 = "Can get insurence?"

print(question1, " => ", mod_chk.check_moderation(input_message=question1))
print(question2, " => ", mod_chk.check_moderation(input_message=question2))

print()

print(question1, " => ", int_cnf.check_intent(input_message=question1))
print(question2, " => ", int_cnf.check_intent(input_message=question2))
