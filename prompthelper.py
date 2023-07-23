"""
# Pre-processes and labels input/output data for the SFT dataset
"""
import re
import sys

# Constants mostly for reference
START_OF_SENTENCE = '|<sos>|' # start of sentence, starts reading for answer
PROMPT_START = '|<ps>|' # start of a prompt 
PROMPT_END = '|<pe>|' # prompt end, stops reading for answer
END_OF_SENTENCE = '|<eos>|' # end of sentence, keep conversation going
START_OF_CONVERSATION = '|<soc>|' # start of a 'Conversation', a set of prompts and answers
END_OF_CONVERSATION = '|<eoc>|' # start of a 'Conversation', a set of prompts and answers

def v1():
    """
    First version of data pre-processing, requires manual entering of prompts 
    """

    prompts_start = 0
    new_prompts = 0

    print('INSTRUCTIONS: ')
    """
    print(f'This program will prompt you for two parts of a conversation: a PROMPT and ANSWER. Use {PROMPT_END} to finish entering' + 
          f' your prompt and {END_OF_CONVERSATION} to finish entering the answer and stop the conversation. {END_OF_SENTENCE} ' + 
          f'can be entered in place of {END_OF_CONVERSATION} to continue the conversation with more prompts and answers as necessary. ')
    """
    print(f'This program will prompt you for two parts of a conversation: a PROMPT and ANSWER. Finish entering prompt and then ' + 
          f'enter {END_OF_CONVERSATION} to finish entering the answer and stop the conversation. {END_OF_SENTENCE} ' + 
          f'can be entered in place of {END_OF_CONVERSATION} to continue the conversation with more prompts and answers as necessary. ')
    print()

    try:
        with open('prompts.txt', 'rb') as f:
            contents = f.read()
            contents = contents.decode(encoding='utf-8')
        prompts_start = len(contents.split('|<eoc>|')) - 1

        # Main loop
        while True:

            with open('prompts.txt', 'ab') as f:

                conversation = START_OF_CONVERSATION
                master_cursor = 'foo'
                print('----- START CONVERSATION -----')
                # Allow multiple prompts for one conversation
                while master_cursor != END_OF_CONVERSATION:
                    
                    prompt = ''
                    answer = ''
                    cursor = input('Prompt: ')
                    
                    # Originally this was the code to check if the prompt sequence had any newline chars
                    """
                    while cursor != PROMPT_END:
                        prompt += cursor 
                        cursor = input()
                        if cursor != PROMPT_END:
                            answer += '\n'
                    """
                    prompt += cursor

                    cursor = input('Desired answer: ')
                    while cursor != END_OF_SENTENCE and cursor != END_OF_CONVERSATION:
                        answer += cursor 
                        cursor = input()
                        if cursor != END_OF_SENTENCE and cursor != END_OF_CONVERSATION:
                            answer += '\n'

                    # Add this prompt to conversation
                    conversation += PROMPT_START
                    conversation += prompt 
                    conversation += '|<pe>|' # += PROMPT_END

                    # Add this answer to conversation
                    conversation += START_OF_SENTENCE
                    conversation += answer
                    conversation = conversation.rstrip('\n')
                    conversation += '|<eos>|' # += END_OF_SENTENCE

                    master_cursor = cursor
                
                conversation += '|<eoc>|' # += END_OF_CONVERSATION
                conversation = '\n' + conversation
                conversation = conversation.encode(encoding='utf-8')

                print('----- END CONVERSATION -----')
                print('\nFULL CONVERSATION: MAKE SURE TO SANITY CHECK !!! ')
                print(conversation)
                print()

                f.write(conversation)
                new_prompts += 1

    except KeyboardInterrupt:
        print()
        print('----- SESSION RESULTS ----- ')
        print(f'OLD PROMPTS: \t{str(prompts_start)}')
        print(f'NEW PROMPTS: \t{str(new_prompts)}')
        print(f'TOTAL PROMPTS: \t{str(prompts_start + new_prompts)}')
    finally:
        if not f.closed:
            f.close()

def v2(debug: bool):
    """
    Second version of data pre-processing from a text file. Still will require manual input to mark start and stop between conversation.
    Requires text from character ai to be put into something like Google Docs and all the italicized text removed before copying into the 
    c_ai.txt file. (I'm really dumb)
    """
    with open("c_ai.txt", 'rb') as f:
        text = f.read()
    text = text.decode(encoding='utf-8')
    m1 = r'([\r\n]*hiyahhhhh\r\n|[\r\n]*Mona\r\nc.ai\r\n)'
    data = [resp for resp in re.split(m1, text) if re.match(m1, resp) == None and resp != '']

    prompts_start = 0
    new_prompts = 0

    with open('prompts.txt', 'rb') as f:
        contents = f.read()
        contents = contents.decode(encoding='utf-8')
    prompts_start = len(contents.split(END_OF_CONVERSATION)) - 1

    # Check if data is valid
    if len(data) % 2: 
        raise AssertionError('Unequal amount of prompts and responses')
    
    current_conversation = START_OF_CONVERSATION
    i = 0 # Iterator for all tokens

    try:
        while i < len(data):
            print()
            print(f'YOU: \t' + data[i])
            current_conversation += (PROMPT_START + data[i] + PROMPT_END)
            i += 1
            print(f'MONA: \t' + data[i])
            current_conversation += (START_OF_SENTENCE + data[i] + END_OF_SENTENCE)
            i += 1
            
            end = 'foo'
            if i < len(data):
                end = input(f'\nEnd of Conversation? Preview: {data[i][:50] if len(data[i]) > 50 else data[i]} (Y/Empty) ')
                
            
            if end == '':
                # Conversation continues
                pass
            else:
                current_conversation += END_OF_CONVERSATION
                current_conversation = '\n' + current_conversation
                print('FULL CONVERSATION: MAKE SURE TO SANITY CHECK !!!' + current_conversation)
                current_conversation = current_conversation.encode(encoding='utf-8')
                with open('prompts.txt', 'ab') as f:
                    
                    if not debug:
                        f.write(current_conversation)
                    new_prompts += 1

                current_conversation = START_OF_CONVERSATION

    except KeyboardInterrupt:
        
        if debug:
            new_prompts = 0

    finally:
        if not f.closed:
            f.close()
        print()
        print('----- SESSION RESULTS ----- ')
        print(f'OLD PROMPTS: \t{str(prompts_start)}')
        print(f'NEW PROMPTS: \t{str(new_prompts)}')
        print(f'TOTAL PROMPTS: \t{str(prompts_start + new_prompts)}')

if __name__ == '__main__':
    
    if 'debug' in sys.argv:
        v2(True)
    else: 
        v2(False)