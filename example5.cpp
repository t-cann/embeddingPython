#define PY_SSIZE_T_CLEAN
#include <Python.h>
//References:
//https://www.linuxjournal.com/article/8497
//Not finished 
void process_expression(char* filename,
                        int num,
                        char* exp[])
{
    FILE*       exp_file;

    // Initialize a global variable for
    // display of expression results
    PyRun_SimpleString("x = 0");

    // Open and execute the file of
    // functions to be made available
    // to user expressions
    exp_file = fopen(filename, "r");
    PyRun_SimpleFile(exp_file, exp[0]);

    // Iterate through the expressions
    // and execute them
    while(num--) {
        PyRun_SimpleString(*exp++);
        PyRun_SimpleString("print(x)");
    }
}

int main(int argc, char* argv[])
{
    Py_Initialize();

    if(argc != 3) {
        printf("Usage: %s FILENAME EXPRESSION+\n");
        return 1;
    }
    process_expression(argv[1], argc - 1, argv + 2);
    return 0;
}
