#include "fixture.h"
#include "digits_data.h"
void digits_graph(size_t rows,ta_graph *g,ta_node *n,size_t *output) {
    n[0]=(ta_node){TA_INPUT,TA_NONE,TA_NONE,rows,64,NULL,NULL,0,0};
    n[1]=(ta_node){TA_DENSE,0,TA_NONE,rows,32,digits_w0,digits_b0,2048,32};
    n[2]=(ta_node){TA_RELU,1,TA_NONE,rows,32,NULL,NULL,0,0};
    n[3]=(ta_node){TA_DENSE,2,TA_NONE,rows,10,digits_w1,digits_b1,320,10};
    *output=3;*g=(ta_graph){4,1,n,output};
}
const float *digits_inputs(void) { return digits_x; }
const unsigned char *digits_labels(void) { return digits_y; }
size_t digits_weight_bytes(void) { return sizeof(digits_w0)+sizeof(digits_b0)+sizeof(digits_w1)+sizeof(digits_b1); }
