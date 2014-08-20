// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use the general purpose non-linear 
    least squares optimization routines from the dlib C++ Library.

    This example program will demonstrate how these routines can be used for data fitting.
    In particular, we will generate a set of data and then use the least squares  
    routines to infer the parameters of the model which generated the data.
*/


//#include <dlib/matrix.h>
//#include <dlib/optimization.h>
//#include <dlib/numeric_constants.h>
//#include <dlib/numerical_integration.h>
#include <iostream>
#include <fstream>
#include <iomanip>
//#include <vector>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
    
#include <python2.7/Python.h>
    
using namespace std;
//using namespace dlib;


#define REAL(z,i) ((z)[2*(i)])
#define IMAG(z,i) ((z)[2*(i)+1])

// ----------------------------------------------------------------------------------------

//typedef matrix<double,2,1> input_vector;
//typedef matrix<double,3,1> parameter_vector;

// ----------------------------------------------------------------------------------------

// We will use this function to generate data.  It represents a function of 2 variables
// and 3 parameters.   The least squares procedure will be used to infer the values of 
// the 3 parameters based on a set of input/output pairs.

double skew_student(double x, double mu, double sig,  double skew, double  nu){
    return 2.*gsl_ran_tdist_pdf((x-mu)/sig, nu)* \
    gsl_cdf_tdist_P(skew*((x-mu)/sig)*sqrt((nu+1.)/(nu+(x-mu)*(x-mu)/sig/sig)),(nu+1.))/sig;
}

double itegral_student(double x, double mu, double sig,  double skew, double  nu){

    return 2.*gsl_ran_tdist_pdf((x-mu)/sig, nu)* \
    gsl_cdf_tdist_P(skew*((x-mu)/sig)*sqrt((nu+1.)/(nu+(x-mu)*(x-mu)/sig/sig)),(nu+1.))/sig;
}

int search_min(const gsl_vector *v, double val){
    for (int i=0; i < v->size; i++)
        if (val < gsl_vector_get(v, i)) 
            return i; 
}

int search_max(const gsl_vector *v, double val){
    for (int i=0; i < v->size; i++)
        if (val < gsl_vector_get(v, i)){ 
            return i; 
        }
}

void complex_vector_abs(gsl_vector *out, const gsl_vector *re, const gsl_vector *im )
{
    for (int i=0; i < out->size; i++ )
        gsl_vector_set(out, i, sqrt( gsl_vector_get(re,i)*gsl_vector_get(re,i) + gsl_vector_get(im,i)*gsl_vector_get(im,i) ) );
}

gsl_vector * c_imag(gsl_vector_complex &v){
    gsl_vector_view im = gsl_vector_complex_real(v);
    gsl_vector_view im_2 = gsl_vector_subvector_with_stride(&im.vector,1, 2,  v->size/2);
    return   &im_2.vector;     
}



//Destroy original vector arr
void hanning(gsl_vector *arr, gsl_vector *x, double xmin, double xmax, double dw){
    double dw2=dw*0.5;
    double kmin_1=xmin-dw2;
    double kmin_2=xmin+dw2;
    double kmax_1=xmax-dw2;
    double kmax_2=xmax+dw2;
    int window_size= dw/(x->data[1] - x->data[0]);
    int min_w=0;
    int max_w=0;
    
    for (int i=0; i< x->size; i++){
        if (x->data[i] < kmin_1) gsl_vector_set(arr, i, 0.);
        else if ( x->data[i] >= kmin_1 && x->data[i] < kmin_2)
        {
            double wind = cos(0.5*M_PI*min_w/(window_size-1.));
            wind=(1.-wind*wind);
            double val = gsl_vector_get(arr, i)*wind;
            gsl_vector_set(arr, i, val);
            min_w++;
        }
        else if ( x->data[i] > kmax_1 && x->data[i] <= kmax_2)
        {
            double wind = cos(0.5*M_PI*max_w/(window_size-1.));
            wind*=wind;
            double val = gsl_vector_get(arr, i)*wind;
            gsl_vector_set(arr, i, val);
            max_w++;
        }
        else if (x->data[i] > kmax_2) gsl_vector_set(arr, i, 0.);
    }
}

void plot_matplotlib(gsl_matrix *m){
   Py_Initialize();
   PyRun_SimpleString("import pylab");
   PyRun_SimpleString("import numpy as np");
   ofstream in;
   char buf[100];
   in.open("tmp.plt");
   
   size_t rows=m->size1;
   size_t cols=m->size2;
   //cout << cols << "\t" << rows << endl;
   for (int i=0; i<rows; i++)
    {
        for (int j = 0; j < cols; j++){
            in << setprecision(8) << gsl_matrix_get(m,i,j) << "\t";
        }
        in << endl;
   }

   PyRun_SimpleString("data=np.loadtxt('tmp.plt')");
   for (int j = 1; j < cols; j++){
       sprintf(buf, "pylab.plot(data[:,0], data[:,%i], label='%i')", j, j);
       PyRun_SimpleString(buf);
   }
   PyRun_SimpleString("pylab.legend()");   
   PyRun_SimpleString("pylab.show()");
   Py_Exit(0);
   in.close();
}


/*double model (
    const input_vector& input,
    const parameter_vector& params
)
{
    const double p0 = params(0);
    const double p1 = params(1);
    const double p2 = params(2);

    const double i0 = input(0);
    const double i1 = input(1);

    const double temp = p0*i0 + p1*i1 + p2;

    return temp*temp;
}*/

// ----------------------------------------------------------------------------------------

// This function is the "residual" for a least squares problem.   It takes an input/output
// pair and compares it to the output of our model and returns the amount of error.  The idea
// is to find the set of parameters which makes the residual small on all the data pairs.
/*
double residual (
    const std::pair<input_vector, double>& data,
    const parameter_vector& params
)
{
    return model(data.first, params) - data.second;
}

// ----------------------------------------------------------------------------------------

// This function is the derivative of the residual() function with respect to the parameters.
parameter_vector residual_derivative (
    const std::pair<input_vector, double>& data,
    const parameter_vector& params
)
{
    parameter_vector der;

    const double p0 = params(0);
    const double p1 = params(1);
    const double p2 = params(2);

    const double i0 = data.first(0);
    const double i1 = data.first(1);

    const double temp = p0*i0 + p1*i1 + p2;

    der(0) = i0*2*temp;
    der(1) = i1*2*temp;
    der(2) = 2*temp;

    return der;
}

// ----------------------------------------------------------------------------------------
*/
int main()
{
    const int max_mu_size=601;
    const int zero_pad_size=pow(2,14);
    FILE *in;
    in= fopen("../mean.chi_9nm", "r");
    gsl_matrix *e = gsl_matrix_alloc(max_mu_size, 4);
    gsl_vector * kvar=gsl_vector_alloc(max_mu_size);
    gsl_vector * muvar=gsl_vector_alloc(max_mu_size);
    gsl_vector * mu_0pad=gsl_vector_alloc(zero_pad_size);
    gsl_vector * r_0pad=gsl_vector_alloc(zero_pad_size/2); //half of lenght 
    gsl_vector * kvar_0pad=gsl_vector_alloc(zero_pad_size);

    gsl_matrix_fscanf(in, e);
    fclose(in);
    gsl_matrix_get_col(kvar,e,0);
    gsl_matrix_get_col(muvar,e,1);
    gsl_vector_set_zero(mu_0pad);


    double dk=gsl_vector_get (kvar, 1)-gsl_vector_get (kvar, 0);
    double dr=M_PI/float(zero_pad_size-1)/dk;

    for (int i = 0; i < zero_pad_size; i++)
    {
      gsl_vector_set (kvar_0pad, i, dk*i);
    }
    for (int i = 0; i < zero_pad_size/2; i++)
    {
      gsl_vector_set (r_0pad, i, dr*i);
    }
    for (int i = 0; i < max_mu_size; i++)
    {
      gsl_vector_set (mu_0pad, i, gsl_vector_get (muvar, i));
    }


    //cout << gsl_vector_get (muvar, 1) <<"\t" << gsl_vector_get (muvar, 2) << endl;
    //plot_matplotlib(e);

    gsl_vector *mu_widowed=gsl_vector_alloc(zero_pad_size);
    gsl_vector_memcpy (mu_widowed, mu_0pad);
    double kmin=3.5, kmax=18.0, dwk=0.8;
    hanning(mu_widowed, kvar_0pad, kmin, kmax, dwk);


    //FFT transform
    double *data = new double [zero_pad_size] ;
    memcpy(data, mu_widowed->data, zero_pad_size*sizeof(double));
    gsl_fft_real_radix2_transform(data, 1, zero_pad_size);

    gsl_vector_complex *fourier_data = gsl_vector_complex_alloc (zero_pad_size);
    gsl_fft_halfcomplex_radix2_unpack(data, fourier_data->data, 1, zero_pad_size);
   
    cout << data[0] << "\t" << data[1] << "\t" << data[2] << "\t" << endl;
    cout << fourier_data->data[0] << "\t" << fourier_data->data[1] << "\t" << fourier_data->data[2] << "\t" << endl;


    //gsl_complex_packed_array fftR = fourier_data->data;
 
    gsl_vector_view fftR_real = gsl_vector_complex_real(fourier_data);
    gsl_vector *fftR_imag = c_imag(*fourier_data);
    gsl_vector *fftR_abs=gsl_vector_alloc(fourier_data->size / 2);
    complex_vector_abs(fftR_abs, &fftR_real.vector, &fftR_imag.vector );
    
    cout << fftR_abs->data[0] << "\t" << fftR_abs->data[1] << "\t" << fftR_abs->data[2] << "\t" << endl;
    cout << fftR_abs->size<< "\t" << (&fftR_real.vector)->size << "\t" << (&fftR_imag.vector)->size << "\t" << endl;


    //Plotting
    /*
    gsl_matrix *plotting = gsl_matrix_calloc(zero_pad_size, 3);
    gsl_matrix_set_col (plotting, 0, kvar_0pad);
    gsl_matrix_set_col (plotting, 1, mu_0pad);
    gsl_matrix_set_col (plotting, 2, mu_widowed);
    int max_k=search_max(kvar_0pad, 35.);
    int min_k=search_max(kvar_0pad, 1.0);
    gsl_matrix_view plotting_lim = gsl_matrix_submatrix (plotting, min_k, 0, max_k-min_k, 3);
    plot_matplotlib(&plotting_lim.matrix);
    gsl_matrix_free (plotting);
    */

    /*
    gsl_matrix *plotting = gsl_matrix_calloc(zero_pad_size, 2);
    gsl_matrix_set_col (plotting, 0, r_0pad);
    gsl_matrix_set_col (plotting, 1, mu_0pad);
    int max_k=search_max(kvar_0pad, 35.);
    int min_k=search_max(kvar_0pad, 1.0);
    gsl_matrix_view plotting_lim = gsl_matrix_submatrix (plotting, min_k, 0, max_k-min_k, 3);
    plot_matplotlib(&plotting_lim.matrix);
    gsl_matrix_free (plotting);
    */
    
    //for (int i=0; i< 20; i++)
    //    cout << (&fftR_real.vector)->data[i]<<endl;
    

    gsl_matrix *plotting = gsl_matrix_calloc(r_0pad->size/2, 4);
    gsl_matrix_set_col (plotting, 0,  r_0pad);
    gsl_matrix_set_col (plotting, 1,  fftR_abs);
    gsl_matrix_set_col (plotting, 2,  &fftR_real.vector);
    gsl_matrix_set_col (plotting, 3,  &fftR_imag.vector);


    int max_r=search_max(r_0pad, 10.);
    int min_r=search_max(r_0pad, 0.);
    gsl_matrix_view plotting_lim = gsl_matrix_submatrix (plotting, min_r, 0, max_r-min_r, plotting->size2);
    plot_matplotlib(&plotting_lim.matrix);
    //plot_matplotlib(plotting);

    gsl_matrix_free (plotting);
    



    //cout << "Done" << endl;
    //cout << data[1] <<"\t" << data[2] << endl;
    
    //for (int i = 0; i < kvar->size; i++)
    //{
    //    cout << gsl_vector_get (kvar, i) <<"\t" << gsl_vector_get (muvar, i) << endl;
    //}

}