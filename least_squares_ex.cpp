//#include <iostream>
//#include <fstream>
//#include <iomanip>
//#include <vector>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_multifit_nlin.h>


//#include <Eigen/Core>
#include <python2.7/Python.h>
    
//using namespace std;
jmp_buf jumper;

struct student_params { double mu; double sig; double skew; double nu; double lam; double omega; double delta; };
struct inter_path  { gsl_interp_accel *acc; gsl_spline *phc_spline; gsl_spline *mag_spline; gsl_spline *pha_spline; gsl_spline *lam_spline;};
struct fit_params  { double mu; double sig; double skew; double nu; double S02; double kshift; };
struct mu_data_fit { gsl_vector *k; gsl_vector *mu; };

static struct inter_path splines;
const double N = 4.; //coordination number

double skew_student(double x, double mu, double sig,  double skew, double  nu){
    return 2.*gsl_ran_tdist_pdf((x-mu)/sig, nu)* \
    gsl_cdf_tdist_P(skew*((x-mu)/sig)*sqrt((nu+1.)/(nu+(x-mu)*(x-mu)/sig/sig)),(nu+1.))/sig;
}

double itegral_student(double x, void * p){
    //delta=phase_f(k)+phc_f(k); omega= 2.0*k
    struct student_params * params = (struct student_params *)p;
    double mu   = (params->mu   );
    double sig  = (params->sig  );
    double skew = (params->skew );
    double nu   = (params->   nu);
    double lam  = (params->lam);    
    double delta= (params->delta);
    double omega= (params->omega);

    double t = skew_student(x, mu, sig, skew, nu);
    //cout << t << "\t" << t/x/x*exp(-2.*x/lam)*sin(omega*x+delta) <<endl;
    return t/x/x*exp(-2.*x/lam)*sin(omega*x+delta);
}


extern "C" {int compute_itegral(const gsl_vector *k, void * p, gsl_vector *out){
    //delta=phase_f(k)+phc_f(k); omega= 2.0*k
    struct fit_params * fp = (struct fit_params *)p;
    double L=10.0, result, error;

    gsl_function F;
    struct student_params params;
    params.mu=fp->mu; params.sig=fp->sig; params.skew=fp->skew; params.nu=fp->nu;
    F.function = &itegral_student;
    F.params = &params;
    if ( params.nu < -1. )
    {
        gsl_vector_set_zero(out);
        return GSL_SUCCESS;
    } 
    gsl_integration_workspace * w         = gsl_integration_workspace_alloc (10000);
    for (int i=1; i< k->size; i++){
        double kv=gsl_vector_get(k, i);
        params.delta= gsl_spline_eval(splines.pha_spline, kv, splines.acc) + \
                       gsl_spline_eval(splines.phc_spline, kv, splines.acc);
        params.omega=2.*kv;
        params.lam  = gsl_spline_eval(splines.lam_spline, kv, splines.acc);
        //cout << i << " " << kv << " "<< params.mu  << " "<< params.sig << " "<< params.skew << " "<< params.nu  << " "<< params.omega << " "<< params.lam <<endl;
        gsl_integration_qag(&F, 0.1, L, 0., 1e-8, 1000, GSL_INTEG_GAUSS51, w, &result, &error); 
        //gsl_integration_qawo (&F, 0., 0., 1e-7, 100, w, int_table, &result, &error);

        //amp=N*S02*mag_f(k_var)*np.power(k_f(k_var), kweight-1.0)*tmp
        result*=N*(fp->S02)*gsl_spline_eval(splines.mag_spline, kv, splines.acc)/kv;
        //cout << kv << "\t" << result <<endl;
        gsl_vector_set(out, i, result);  
    }
    gsl_integration_workspace_free (w);
    return GSL_SUCCESS;}
}


extern "C" {
    int resudial_itegral(const gsl_vector *in, void * p, gsl_vector *out){
    //delta=phase_f(k)+phc_f(k); omega= 2.0*k
    struct mu_data_fit * mu = (struct mu_data_fit *)p;
    struct fit_params fp = {in->data[0], in->data[1], in->data[2], \
                            in->data[3], in->data[4], in->data[5]} ;

    //cout << "dasdas " << fp.mu  << " "<< fp.sig << " "<< fp.skew << " "<< fp.nu   <<endl;
    for (int i =0; i< in->size; i++){
        printf("%14.5f", gsl_vector_get (in, i)) ;
    }
    printf("\n") ;
    //cout << "1213rw " << gsl_vector_get (in, 0)  << endl; // " "<< guess->data[1] << " "<< guess->data[2] << " "<< guess->data[3]   <<endl;
    //cout << "gggg" << mu->k->data[0] << " "<< mu->k->data[1] << " "<< mu->mu->data[0] << " "<< mu->mu->data[1]   <<endl;
    
    gsl_vector_set_zero(out);
    compute_itegral(mu->k, &fp, out);
    /*
    for (int i =0; i< in->size; i++){
        printf("%10.5f", gsl_vector_get (out, i)) ;
    }
    printf("\n") ;*/
    gsl_vector_set (out, 0, 0.0) ;
    gsl_vector_sub(out, mu->mu);
    //for (int i =0; i< in->size; i++){
    //    printf("%10.5f", gsl_vector_get (out, i)) ;
    //}
    //printf("\n") ;

    return GSL_SUCCESS;}
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

void complex_vector_abs(gsl_vector *out, const gsl_vector *re, const gsl_vector *im ){
    for (int i=0; i < out->size; i++ )
        gsl_vector_set(out, i, sqrt( gsl_vector_get(re,i)*gsl_vector_get(re,i) + gsl_vector_get(im,i)*gsl_vector_get(im,i) ) );
}

void complex_vector_parts(const gsl_vector_complex *in, gsl_vector *re, gsl_vector *im ){
    for (int i=0; i < in->size; i+=2 ){
        gsl_vector_set(re, int(i/2), in->data[i]);
        gsl_vector_set(im, int(i/2), in->data[i+1]);
    }
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
   FILE *in;
   char buf[100];
   in = fopen("tmp.plt","w");
   
   size_t rows=m->size1;
   size_t cols=m->size2;
   //cout << cols << "\t" << rows << endl;
   for (int i=0; i<rows; i++)
    {
        for (int j = 0; j < cols; j++){
            fprintf(in, "%10.6f ", gsl_matrix_get(m,i,j));
        }
        fprintf(in, "\n ");
   }

   PyRun_SimpleString("data=np.loadtxt('tmp.plt')");
   for (int j = 1; j < cols; j++){
       sprintf(buf, "pylab.plot(data[:,0], data[:,%i], label='%i')", j, j);
       PyRun_SimpleString(buf);
   }
   PyRun_SimpleString("pylab.legend()");   
   PyRun_SimpleString("pylab.show()");
   Py_Exit(0);
   fclose(in);
}


int main()
{
    const int max_mu_size=601;
    const int zero_pad_size=pow(2,15);
    FILE *in;
    in= fopen("../mean.chi", "r");
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
    gsl_matrix_free(e);


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
    double *data = (double *) malloc(zero_pad_size*sizeof(double)); 
    //new double [zero_pad_size] ;
    memcpy(data, mu_widowed->data, zero_pad_size*sizeof(double));
    gsl_fft_real_radix2_transform(data, 1, zero_pad_size);

    //Unpack complex vector
    gsl_vector_complex *fourier_data = gsl_vector_complex_alloc (zero_pad_size);
    gsl_fft_halfcomplex_radix2_unpack(data, fourier_data->data, 1, zero_pad_size);
    gsl_vector *fftR_real = gsl_vector_alloc(fourier_data->size/2);
    gsl_vector *fftR_imag = gsl_vector_alloc(fourier_data->size/2);
    gsl_vector *fftR_abs  = gsl_vector_alloc(fourier_data->size/2);
    complex_vector_parts(fourier_data, fftR_real, fftR_imag);
    complex_vector_abs(fftR_abs, fftR_real, fftR_imag);
    
    gsl_vector *first_shell=gsl_vector_alloc(fftR_abs->size);
    gsl_vector_memcpy (first_shell, fftR_abs);
    double rmin=1.0, rmax=2.85, dwr=0.1;
    hanning(first_shell, r_0pad, rmin, rmax, dwr);


    //feff0001.dat
    const int path_lines=68; 
    e = gsl_matrix_alloc(path_lines, 7); 
    gsl_vector * k_p  =gsl_vector_alloc(path_lines);
    gsl_vector * phc_p=gsl_vector_alloc(path_lines);
    gsl_vector * mag_p=gsl_vector_alloc(path_lines);
    gsl_vector * pha_p=gsl_vector_alloc(path_lines);
    gsl_vector * lam_p=gsl_vector_alloc(path_lines);
    
    in= fopen("../FEFF/feff0001.dat", "r");
    gsl_matrix_fscanf(in, e);
    fclose(in);
    
    gsl_matrix_get_col(k_p  ,e,0);
    gsl_matrix_get_col(phc_p,e,1);
    gsl_matrix_get_col(mag_p,e,2);
    gsl_matrix_get_col(pha_p,e,3);
    gsl_matrix_get_col(lam_p,e,5);
    gsl_matrix_free(e);

    gsl_interp_accel *acc = gsl_interp_accel_alloc ();
    gsl_spline *k_spline   = gsl_spline_alloc (gsl_interp_cspline, path_lines);
    gsl_spline *phc_spline = gsl_spline_alloc (gsl_interp_cspline, path_lines);
    gsl_spline *mag_spline = gsl_spline_alloc (gsl_interp_cspline, path_lines);
    gsl_spline *pha_spline = gsl_spline_alloc (gsl_interp_cspline, path_lines);
    gsl_spline *lam_spline = gsl_spline_alloc (gsl_interp_cspline, path_lines);

    gsl_spline_init (k_spline  , k_p->data, k_p->data  , path_lines);
    gsl_spline_init (phc_spline, k_p->data, phc_p->data, path_lines);
    gsl_spline_init (mag_spline, k_p->data, mag_p->data, path_lines);
    gsl_spline_init (pha_spline, k_p->data, pha_p->data, path_lines);
    gsl_spline_init (lam_spline, k_p->data, lam_p->data, path_lines);


    gsl_vector * mu_p  =gsl_vector_alloc(path_lines);

    //struct fit_params { student_params t; double kshift; double S02; double N; inter_path splines; };
    //student_params t   = {2.45681867, 0.02776907, -21.28920008, 9.44741797, 0.0, 0.0, 0.0};

    splines.acc=acc; splines.phc_spline=phc_spline; splines.mag_spline=mag_spline;
    splines.pha_spline=pha_spline; splines.lam_spline=lam_spline;
    
    
    fit_params fp = { 2.45681867, 0.02776907, -21.28920008, 9.44741797, 1.0, 0.0};
    compute_itegral(k_p, &fp, mu_p);

    mu_data_fit params = { k_p, mu_p};

    // initialize the solver
    size_t Nparams=6;
    gsl_vector *guess0 = gsl_vector_alloc(Nparams);

    gsl_vector_set(guess0, 0, 2.45);
    gsl_vector_set(guess0, 1, 0.01);
    gsl_vector_set(guess0, 2, 10.0);
    gsl_vector_set(guess0, 3,  5.0);
    gsl_vector_set(guess0, 4,  1.0);
    gsl_vector_set(guess0, 5,  0.0);


    gsl_multifit_function_fdf fit_mu_k;
    fit_mu_k.f = &resudial_itegral;
    fit_mu_k.n = path_lines;
    fit_mu_k.p = Nparams;
    fit_mu_k.params = &params;
    fit_mu_k.df = NULL;
    fit_mu_k.fdf = NULL;




    gsl_multifit_fdfsolver *solver = gsl_multifit_fdfsolver_alloc(gsl_multifit_fdfsolver_lmsder, path_lines, Nparams);
    gsl_multifit_fdfsolver_set(solver, &fit_mu_k, guess0);

    size_t iter=0, status;
    do{
        iter++;
        //cout << solver->x->data[0] << " " << solver->x->data[1] <<endl;
        status = gsl_multifit_fdfsolver_iterate (solver);
        printf("%12.4f %12.4f %12.4f\n", solver->J->data[0,0], solver->J->data[1,1], solver->J->data[2,2] );
        //gsl_multifit_fdfsolver_dif_df  (k_p, &fit_mu_k, mu_p, solver->J);
        //gsl_multifit_fdfsolver_dif_fdf (k_p, &fit_mu_k, mu_p, solver->J);
        printf ("status = %s\n", gsl_strerror (status));
        //print_state (iter, solver);
        if (status) break;
        status = gsl_multifit_test_delta (solver->dx, solver->x, 1e-4, 1e-4);
    }while (status == GSL_CONTINUE && iter < 500);



    //cout << gsl_spline_eval (k_spline, 1.333, acc) << endl;
    //cout << gsl_spline_eval (phc_spline, 1.333, acc) << endl;


    //cout << data[0] << "\t" << data[1] << "\t" << data[2] << "\t" << endl;
    //cout << fourier_data->data[0] << "\t" << fourier_data->data[1] << "\t" << fourier_data->data[2] << "\t" << endl;

   
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
  
    gsl_matrix *plotting = gsl_matrix_calloc(r_0pad->size, 5);
    gsl_matrix_set_col (plotting, 0,  r_0pad);
    gsl_matrix_set_col (plotting, 1,  fftR_abs);
    gsl_matrix_set_col (plotting, 2,  fftR_real);
    gsl_matrix_set_col (plotting, 3,  fftR_imag);
    gsl_matrix_set_col (plotting, 4,  first_shell);
    
    int min_r=search_max(r_0pad, 0.);
    int max_r=search_max(r_0pad, 5.);
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