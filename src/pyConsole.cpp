#define NPY_NO_DEPRECATED_API  NPY_1_9_API_VERSION //TODO Resolve issue with old api files but frames python os ??? https://numpy.org/devdocs/reference/c-api/deprecations.html
#include "pyConsole.h"
#include <numpy/arrayobject.h>

// References:
//https://ubuverse.com/embedding-the-python-interpreter-in-a-qt-application/
//http://mateusz.loskot.net/post/2011/12/01/python-sys-stdout-redirection-in-cpp/
//https://codereview.stackexchange.com/questions/92266/sending-a-c-array-to-python-numpy-and-back/92353#92353
//https://ubuverse.com/embedding-the-python-interpreter-in-a-qt-application/
//https://docs.scipy.org/doc//numpy-1.15.0/reference/c-api.html
//https://docs.python.org/3/extending/embedding.html


//Notes:
//Sending data from C++ to Python
//Copy vs Move
//Deep vs Shallow Copy
//Type of C Array vs Python Variable 
//As numpy is C Accelerated  might help
//Tiobrowse data
//Simple types of data - Strings , Ints and Doubles

/**
 * @brief Construct a new py Console::py Console object
 * 
 */
pyConsole::pyConsole()
{
    //program = Py_DecodeLocale((char*)argv[0], NULL); //TODO
    program = Py_DecodeLocale("embeddingPythonConsole", NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }
    Py_SetProgramName(program);  // optional but recommended 

    std::string stdOutErr =
            "import sys, os\n"
            "sys.path.append(os.getcwd())\n"
            "\n"
            "class CatchOutErr:\n"
            "    def __init__(self):\n"
            "        self.value = ''\n"
            "    def write(self, txt):\n"
            "        self.value += txt\n"
            "\n"
            "catchOutErr = CatchOutErr()\n"
            "oldstdout = sys.stdout\n"
            "sys.stdout = catchOutErr\n"
            "oldstderr = sys.stderr\n"
            "sys.stderr = catchOutErr\n"
            ; //this is python code to redirect stdouts/stderr

    Py_Initialize();
    
    // if(_import_array()<0){
    // }
    //PyRun_SimpleString("print('>>> Start of Python Output / Constructor PyConsole')");

    pModule = PyImport_AddModule("__main__"); //create main module
    PyRun_SimpleString(stdOutErr.c_str()); //invoke code to redirect
    catcher = PyObject_GetAttrString(pModule,"catchOutErr"); //get our catchOutErr created above  
    PyErr_Print(); //make python print any errors
    output = PyObject_GetAttrString(catcher,"value"); //get the stdout and stderr from our catchOutErr object

    //Uncomment to Test Functionality (TODO Make Unit Test)
    //displayDateandTime();

//     stringtoConsole("name", "Thomas");
//     inttoConsole();
//     doubletoConsole();
//     arraytoConsole();
}

/**
 * @brief Destroy the py Console::py Console object
 * 
 */
pyConsole::~pyConsole()
{
    PyRun_SimpleString("sys.stdout = oldstdout");

    //PyRun_SimpleString("print('<<< End of Python Output / Deconstructor PyConsole')");
    //Py_FinalizeEx(); //Notice that Py_FinalizeEx() does not free all memory allocated by the Python interpreter, e.g. memory allocated by extension modules currently cannot be released.
    
    if (Py_FinalizeEx() < 0) {
        exit(120);
    }
    Py_Finalize();
    PyMem_RawFree(program);
}

/**
 * @brief Run a C++ String (const char *) in Python e.g. if command = "print('hello')" -> hello gets sent to stdout. 
 * 
 * @param command 
 */
void pyConsole::runString(const char *command){
    PyRun_SimpleString(command);
}

// /**
//  * @brief Takes in a QString which is run in python and the output is captured and returned.
//  * 
//  * @param command 
//  * @return QString 
//  */
// QString pyConsole::pyRun(QString command){
//     QString input =  command; //"\"" + command + "\"";
//     PyRun_SimpleString("catchOutErr.value = ''");
//     PyRun_SimpleString(input.toStdString().c_str());

//     catcher = PyObject_GetAttrString(pModule,"catchOutErr"); //get our catchOutErr created above
//     PyErr_Print(); //make python print any errors

//     output = PyObject_GetAttrString(catcher,"value"); //get the stdout and stderr from our catchOutErr object


//     QString outstring = pyConsole::ObjectToString(output);
//     return outstring;
// }

// /**
//  * @brief Returns a QString if a PyObject is a Unicode String.
//  * 
//  * @param poVal 
//  * @return QString 
//  */
// QString pyConsole::ObjectToString(PyObject *poVal)
// {
//     QString val;

//     if(poVal != NULL)
//     {
//         if(PyUnicode_Check(poVal))
//         {
//             // Convert Python Unicode object to UTF8 and return pointer to buffer
//             const char *str = PyUnicode_AsUTF8(poVal);

//             if(!hasError())
//             {
//                 val = QString::fromUtf8(str);
//             }
//         }

//         // Release reference to object
//         Py_XDECREF(poVal);
//     }

//     return val;
// }

/**
 * @brief Check is Python has error if so outputs the error to stderr and clear error indicator.
 * 
 * @return true 
 * @return false 
 */
bool pyConsole::hasError()
{
    bool error = false;

    if(PyErr_Occurred())
    {
        // Output error to stderr and clear error indicator
        PyErr_Print();
        error = true;
    }

    return error;
}


/**
 * @brief Function to Test Functionality of Python Interpreter by Outputing Todays Date and Time.
 * 
 */
void pyConsole::displayDateandTime(){
    
    PyRun_SimpleString(
        "from time import time,ctime\n"
        "print('Today is', ctime(time()))\n"
    );

    // PyRun_SimpleString("1+3"); //no output 
    // //PyRun_SimpleString("print(1+1)"); //this is ok stdout
    // //PyRun_SimpleString("1+a"); //this creates an error
    // //qDebug().noquote() <<"Catcher Output:\n\n" + pyConsole::ObjectToString(output).toUtf8(); // In Unicode format \n new line charaters how to remove?
    // //qDebug().noquote() <<  pyConsole::pyRun("print('hello')");

}

// /**
//  * @brief 
//  * 
//  */
// void pyConsole::stringtoConsole(QString valName ,QString value){
//     PyObject *m, *v;
//     m= pModule;
//     v = Py_BuildValue("s",value.toStdString().c_str());
//     PyObject_SetAttrString(m, valName.toStdString().c_str(), v);
// }

// /**
//  * @brief 
//  * 
//  */
// void pyConsole::inttoConsole(QString valName,int  value){
//     PyObject *m, *v;
//     m= pModule;
//     v = Py_BuildValue("i",value);
//     PyObject_SetAttrString(m, valName.toStdString().c_str(), v);
// }

// /**
//  * @brief 
//  * 
//  */
// void pyConsole::doubletoConsole(QString valName, double  value){
//     PyObject *m, *v;
//     m= pModule;
//     v = Py_BuildValue("d",value);
//     PyObject_SetAttrString(m, valName.toStdString().c_str(), v);
// }

// /**
//  * @brief 
//  * 
//  */
// void pyConsole::arraytoConsole(QString valName){
//     bool debug = true;
//     PyObject *m, *v;
//     m= pModule;

//     if(debug){
    

    
//     // Build the 2D array in C++
//     const int SIZE = 10;
//     npy_intp dims[2]{SIZE, SIZE};
//     const int ND = 2;
//     long double(*c_arr)[SIZE]{ new long double[SIZE][SIZE] };
//     for (int i = 0; i < SIZE; i++)
//         for (int j = 0; j < SIZE; j++)
//             c_arr[i][j] = i * SIZE + j;
    
//     PyObject *pArray = PyArray_SimpleNewFromData(ND, dims, NPY_LONGDOUBLE, reinterpret_cast<void*>(c_arr));
//     PyArrayObject *np_arr = reinterpret_cast<PyArrayObject*>(pArray);
//     v = Py_BuildValue("O", np_arr);
//     PyObject_SetAttrString(m, valName.toStdString().c_str(), v);
    
//     }
// }