symnmf: symnmf.o
	@echo "Building symnmf"
	@gcc -ansi -Wall -Wextra -Werror -pedantic-errors -o symnmf symnmf.o -lm

symnmf.o: symnmf.c symnmf.h
	@echo "Compiling symnmf.c"
	@gcc -ansi -Wall -Wextra -Werror -pedantic-errors -c symnmf.c

clean:
	@echo "Cleaning up"
	@rm -f *.o symnmf