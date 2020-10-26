#define PY_SSIZE_T_CLEAN
#include <Python.h>
//References: https://www.linuxjournal.com/article/8497


/**
 * @brief Opens a file and runs a file, loop over command certain number of times.
 * 
 * @param filename 
 * @param num 
 * @param exp 
 */
void process_expression(char* filename, int num, char* command) //, char* exp)
{
    FILE*       exp_file;

    // Initialize a global variable for
    // display of expression results
    PyRun_SimpleString("x = 0");

    // Open and execute the file of
    // functions to be made available
    // to user expressions
    exp_file = fopen(filename, "r");
    PyRun_SimpleFile(exp_file, "exp"); // Don't understand relevance of second argument here. Does not effect result if changed. 

    // Iterate through the expressions and execute them
    while(num--) {
        PyRun_SimpleString(command);
        PyRun_SimpleString("print(x)");
    }
}

/**
 * @brief Run a file and runs a command. 
 * 
 * e.g. ./example3 helloworld.py x+=10 
 *   Hello World!
 *   10
 *   20
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char* argv[])
{
    Py_Initialize();

    if(argc != 3) {
        printf("Usage: %s FILENAME EXPRESSION+\n", argv[0]);
        return 1;
    }
    process_expression(argv[1], argc - 1, argv[2]);
    return 0;
}
