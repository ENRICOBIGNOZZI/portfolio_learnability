/* Same Matern-3/2 managed features; scalar sums, no fast-math or jitter.
 * Workers partition anchors only, so arithmetic order does not depend on
 * thread scheduling. Prefix stock counts preserve the exact 1/N convention.
 */
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

typedef struct {int n,p,k,ncuts,start,end; const double *z,*anchors,*r; const int *cuts; double *out;} job;
static void *calculate(void *pointer) {
    job *a=(job *)pointer;
    const double scale=sqrt(3.0);
    double *radius=malloc(sizeof(double)*a->n);
    double *exponential=malloc(sizeof(double)*a->n);
    for(int j=a->start;j<a->end;j++) {
        for(int i=0;i<a->n;i++) {
            double distance=0;
            for(int d=0;d<6;d++) {double delta=a->z[i*6+d]-a->anchors[j*6+d];distance+=delta*delta;}
            radius[i]=scale*sqrt(distance);exponential[i]=-radius[i];
        }
#ifdef __APPLE__
        vvexp(exponential,exponential,&a->n);
#else
        for(int i=0;i<a->n;i++)exponential[i]=exp(exponential[i]);
#endif
        double sums[3]={0,0,0};int cut=0;
        for(int i=0;i<a->n;i++) {
            double kernel=(1.0+radius[i])*exponential[i];
            for(int c=0;c<a->k;c++)sums[c]+=a->r[i*a->k+c]*kernel;
            if(cut<a->ncuts && i+1==a->cuts[cut]) {
                for(int c=0;c<a->k;c++)a->out[(cut*a->k+c)*a->p+j]=sums[c]/a->cuts[cut];
                cut++;
            }
        }
    }
    free(radius);free(exponential);
    return NULL;
}
int managed(int n,int p,int k,int ncuts,const double *z,const double *anchors,const double *r,const int *cuts,double *out,int threads) {
    if(k<1 || k>3 || ncuts<1 || threads<1 || threads>16)return 1;
    pthread_t workers[16];job jobs[16];
    for(int t=0;t<threads;t++) {
        jobs[t]=(job){n,p,k,ncuts,p*t/threads,p*(t+1)/threads,z,anchors,r,cuts,out};
        if(pthread_create(&workers[t],NULL,calculate,&jobs[t]))return 2;
    }
    for(int t=0;t<threads;t++)pthread_join(workers[t],NULL);
    return 0;
}
