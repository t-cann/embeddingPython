#include <numpy/ndarraytypes.h>

enum TIO_DATA = {
    TIO_DATATYPE_NULL = 0,
    TIO_SHORT, 
    TIO_USHORT, 
    TIO_INT,
    TIO_UINT,
    TIO_LONG,
    TIO_ULONG,
    TIO_LLONG,
    TIO_ULLONG,
    TIO_FLOAT,
    TIO_DOUBLE,
    TIO_LDOUBLE,
    TIO_LOGICAL,
    TIO_CHAR,
    TIO_UCHAR,
    TIO_STRING
};

int getNPYType(TIO_DATA tioType)
{   
    NPY_TYPES np_type;

    switch (tioType) {
    case TIO_DATATYPE_NULL:
            m_data = nullptr;
            break;
        case TIO_SHORT:
            m_data = new short[m_arrayLength];
            break;
        case TIO_USHORT:
            m_data = new unsigned short[m_arrayLength];
            break;
        case TIO_INT:
            m_data = new int[m_arrayLength];
            break;
        case TIO_UINT:
            m_data = new unsigned int[m_arrayLength];
            break;
        case TIO_LONG:
            m_data = new long[m_arrayLength];
            break;
        case TIO_ULONG:
            m_data = new unsigned[m_arrayLength];
            break;
        case TIO_LLONG:
            m_data = new long long[m_arrayLength];
            break;
        case TIO_ULLONG:
            m_data = new unsigned long long[m_arrayLength];
            break;
        case TIO_FLOAT:
            m_data = new float[m_arrayLength];
            break;
        case TIO_DOUBLE:
            m_data = new double[m_arrayLength];
            break;
        case TIO_LDOUBLE:
            m_data = new long double[m_arrayLength];
            break;
        case TIO_LOGICAL:
            m_data = new unsigned int[m_arrayLength];
            break;
        case TIO_CHAR:
            m_data = new char[m_arrayLength];
            break;
        case TIO_UCHAR:
            m_data = new unsigned char[m_arrayLength];
            break;
        case TIO_STRING:
            m_data = new char[m_arrayLength][TIO_STRLEN];
            break;
        default:
            // ErrorDialog("Error: unexpected data type");
            m_data = nullptr;
    }
}
