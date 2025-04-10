#pragma once

#include <cstdlib>
#include <iostream>
#include <string>
#include <cmath>
#include <ctime>
#include <fstream>
#include <vector>
#include <boost/array.hpp>
#include <boost/range.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/multi_array.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include "eigen3/Eigen/Eigen"
#include <sys/time.h>


using std::fabs;
using std::cerr;
using std::cout;
using std::endl;
using std::floor;
using std::pow;
using std::sqrt;
using std::string;
using std::vector;
using std::ceil;
using std::exp;
using std::floor;
using std::log;
using std::ostream;
using std::ofstream;
using std::exp;
using std::log;

using boost::mt19937;
using boost::uniform_real;
using boost::variate_generator;
#include "LnDouble.hpp"
#define BLOCK_SIZE 256 

namespace HMM
{
    template <typename T> class ModelStateTp
    {
	public:

	    ModelStateTp(){};
	    virtual ~ModelStateTp();

	    void FillModelState(int StatesSize,int OupSize);
	    void FillRandom();

	    void SetTransValue(int id, T prob);
	    void SetEmitsValue(int id, T prob);

	    T TransValue(int id) const;
	    T EmitsValue(int id) const;

	    int GetStatesSize() const;
	    int GetOutputsSize() const;

	    boost::numeric::ublas::vector<T> VecTransitions;
	    boost::numeric::ublas::vector<T> VecEmissions;

	private:
	    int pvtStatesSize;
	    int pvtOutputsSize;
	    static mt19937 gen;
	    static variate_generator<mt19937, uniform_real<> > rand;
	    static unsigned int GenRandomSeed();
    };

    template <typename T> inline T ModelStateTp<T>::TransValue(int id) const{ return VecTransitions[id]; }
    template <typename T> inline T ModelStateTp<T>::EmitsValue(int id) const{ return VecEmissions[id]; }
    template <typename T> inline void ModelStateTp<T>::SetTransValue(int id, T prob){ VecTransitions[id] = prob; }
    template <typename T> inline void ModelStateTp<T>::SetEmitsValue(int id, T prob){ VecEmissions[id] = prob; }
    template <typename T> inline int ModelStateTp<T>::GetStatesSize() const {return pvtStatesSize;}
    template <typename T> inline int ModelStateTp<T>::GetOutputsSize() const {return pvtOutputsSize;}
    template <typename T> mt19937 ModelStateTp<T>::gen( static_cast<unsigned int>(ModelStateTp::GenRandomSeed()));
    template <typename T> variate_generator<mt19937, uniform_real<> > ModelStateTp<T>::rand(gen, uniform_real<double>(0.0, 1.0));
    template <typename T> void ModelStateTp<T>::FillModelState(int StatesSize, int OupSize){
	pvtStatesSize = StatesSize;
	pvtOutputsSize = OupSize;
	VecTransitions.resize(StatesSize);
	VecEmissions.resize(OupSize);
    }
    template <typename T> ModelStateTp<T>::~ModelStateTp()
    {
	VecTransitions.resize(0);
	VecEmissions.resize(0);
    }
    template <typename T> unsigned int ModelStateTp<T>::GenRandomSeed()
    {
	timeval curTime;
	gettimeofday(&curTime, NULL);
	//return static_cast<unsigned int>(curTime.tv_usec);
	return static_cast<unsigned int>(10);
    }
    template <typename T> void ModelStateTp<T>::FillRandom()
    {
	T count = 0;
	for (int i = 0; i < pvtStatesSize; i++){
	    VecTransitions[i] = rand();
	    count += VecTransitions[i];
	}
	for (int i = 0; i < pvtStatesSize; i++){
	    VecTransitions[i] /= count;
	}
	count = 0;
	for (int i = 0; i < pvtOutputsSize; i++){
	    VecEmissions[i] = rand();
	    count += VecEmissions[i];
	}
	for (int i = 0; i < pvtOutputsSize; i++){
	    VecEmissions[i] /= count;
	}
    }


    class ModelSpace
    {
	public:
	    explicit ModelSpace();
	    int learn(int iterations = 100);
	    void ReadFromFile(const std::string &FileName,boost::multi_array<boost::numeric::ublas::vector<int>, 1> & data);
	    void ShowResults(boost::multi_array<boost::numeric::ublas::vector<int>, 1>& ViterbiStates, boost::multi_array<boost::numeric::ublas::vector<int>, 1> &OupSerials, std::string OupFileName,int IndependentSerialsSize,boost::numeric::ublas::vector<unsigned long> &SerialsLengthArray);
	    template <typename T> static void ShowHMM(std::ostream& os, boost::multi_array<std::unique_ptr<ModelStateTp<T>>, 1>&states, int StatesSize, int OupSize, boost::numeric::ublas::vector<T>& LeadingProb);
	    void ShowMarkStates(int IndependentSerialsSize,int OupSize,int StatesSize,boost::multi_array<std::unique_ptr<ModelStateTp<double>>, 1>&states, boost::numeric::ublas::vector<unsigned long>& SerialLength, boost::multi_array<boost::numeric::ublas::vector<int>, 1>& data,boost::multi_array<boost::numeric::ublas::vector<int>, 1> &calledStates );
	    void ShowMarkStatesPerObservation(int IndependentSerialsSize,int OupSize,int StatesSize,boost::multi_array<std::unique_ptr<ModelStateTp<double>>, 1>&states, boost::numeric::ublas::vector<unsigned long>& SerialLength, boost::multi_array<boost::numeric::ublas::vector<int>, 1>& data,boost::multi_array<boost::numeric::ublas::vector<int>, 1> &calledStates );
    };

    ModelSpace::ModelSpace(){}

    void ModelSpace::ShowMarkStatesPerObservation(int IndependentSerialsSize,int OupSize,int StatesSize,boost::multi_array<std::unique_ptr<ModelStateTp<double>>, 1>&states, boost::numeric::ublas::vector<unsigned long>& SerialLength, boost::multi_array<boost::numeric::ublas::vector<int>, 1>& data,boost::multi_array<boost::numeric::ublas::vector<int>, 1>& calledStates )
    {

	boost::numeric::ublas::matrix<long double> ExpOupFreq;
	ExpOupFreq.resize(StatesSize,OupSize);
	ExpOupFreq.clear();

	for (int i = 0; i < StatesSize; i++)
	{
	    for (int k = 0; k < OupSize; k++)
	    {
		ExpOupFreq(i,k) = states[i]->EmitsValue(k);
	    }
	}

	boost::numeric::ublas::matrix<long double> ObsOupFreq;
	ObsOupFreq.resize(StatesSize,OupSize);
	ObsOupFreq.clear();

	boost::numeric::ublas::vector<unsigned long> OupArray;
	OupArray.resize(OupSize);
	OupArray.clear();

	boost::numeric::ublas::vector<unsigned long> StateFreq;
	StateFreq.resize(StatesSize);
	StateFreq.clear();

	for (int SerialsID = 0; SerialsID < IndependentSerialsSize; SerialsID++)
	{
	    boost::numeric::ublas::vector<unsigned long> OupPerSerials;
	    OupPerSerials.resize(OupSize);
	    OupPerSerials.clear();

	    boost::numeric::ublas::vector<unsigned long> StateFreqPerSerials;
	    StateFreqPerSerials.resize(StatesSize);
	    StateFreqPerSerials.clear();

	    for (unsigned long loc = 0; loc < SerialLength[SerialsID]; loc++)
	    {
		OupPerSerials[data[SerialsID][loc]]++;
		ObsOupFreq(calledStates[SerialsID][loc],data[SerialsID][loc])++;
		StateFreqPerSerials[calledStates[SerialsID][loc]]++;
	    }
	    for (int j = 0; j < OupSize; j++)
	    {
		OupArray[j] += OupPerSerials[j];
	    }
	    for (int j = 0; j < StatesSize; j++)
	    {
		StateFreq[j] += StateFreqPerSerials[j];
	    }
	    OupPerSerials.resize(0);
	    StateFreqPerSerials.resize(0);
	}

	cout << "State frequencies:" << endl;
	for (int i = 0; i < StatesSize; i++)
	{
	    cout << StateFreq[i] << endl;
	    for (int j = 0; j < OupSize; j++)
	    {
		ObsOupFreq(i,j) /= ((long double) StateFreq[i]);
	    }
	}

	for (int i = 0; i < StatesSize; i++)
	{
	    cout << "State " << i << ":" << endl;
	    for (int j = 0; j < OupSize; j++)
	    {
		cout << ExpOupFreq(i,j) << "\t" << ObsOupFreq(i,j) << endl;
	    }
	}

	for (int i = 0; i < OupSize; i++)
	{
	    cout << "Total amount of mark " << i << ": " << OupArray[i] << endl;
	}

	OupArray.resize(0);
	StateFreq.resize(0);
	ExpOupFreq.resize(0,0);
	ObsOupFreq.resize(0,0);
    }


    void ModelSpace::ShowMarkStates(int IndependentSerialsSize,int OupSize,int StatesSize,boost::multi_array<std::unique_ptr<ModelStateTp<double>>, 1>&states, boost::numeric::ublas::vector<unsigned long>& SerialLength, boost::multi_array<boost::numeric::ublas::vector<int>, 1>& data,boost::multi_array<boost::numeric::ublas::vector<int>, 1> &calledStates )
    {

	int LogOupSize = log2(OupSize);
	boost::numeric::ublas::matrix<long double> ExpOupFreq;
	ExpOupFreq.resize(StatesSize,OupSize);
	ExpOupFreq.clear();

	for (int i = 0; i < StatesSize; i++)
	{
	    for (int k = 0; k < LogOupSize; k++)
	    {
		ExpOupFreq(i,k) = 0.0l;
		for (int j = (1 << k); j < OupSize; j++)
		{
		    if (((1 << k) & j) > 0)
		    {
			ExpOupFreq(i,k) += states[i]->EmitsValue(j);
		    }
		}
	    }
	}

	boost::numeric::ublas::matrix<long double> ObsOupFreq;
	ObsOupFreq.resize(StatesSize,OupSize);
	ObsOupFreq.clear();

	boost::numeric::ublas::vector<unsigned long> OupArray;
	OupArray.resize(LogOupSize);
	OupArray.clear();

	boost::numeric::ublas::vector<unsigned long> StateFreq;
	StateFreq.resize(StatesSize);
	StateFreq.clear();

	for (int SerialsID = 0; SerialsID < IndependentSerialsSize; SerialsID++)
	{
	    boost::numeric::ublas::vector<unsigned long> OupPerSerials;
	    OupPerSerials.resize(LogOupSize);
	    OupPerSerials.clear();

	    boost::numeric::ublas::vector<unsigned long> StateFreqPerSerials;
	    StateFreqPerSerials.resize(StatesSize);
	    StateFreqPerSerials.clear();

	    for (unsigned long loc = 0; loc < SerialLength[SerialsID]; loc++)
	    {
		for (int k = 0; k < LogOupSize; k++)
		{
		    if (((1 << k) & data[SerialsID][loc]) > 0)
		    {
			OupPerSerials[k]++;
			ObsOupFreq(calledStates[SerialsID][loc],k)++;
		    }
		}
		StateFreqPerSerials[calledStates[SerialsID][loc]]++;
	    }
	    for (int j = 0; j < LogOupSize; j++)
	    {
		OupArray[j] += OupPerSerials[j];
	    }
	    for (int j = 0; j < StatesSize; j++)
	    {
		StateFreq[j] += StateFreqPerSerials[j];
	    }
	    OupPerSerials.resize(0);
	    StateFreqPerSerials.resize(0);
	}

	cout << "State frequencies:" << endl;
	for (int i = 0; i < StatesSize; i++)
	{
	    for (int j = 0; j < LogOupSize; j++)
	    {
		ObsOupFreq(i,j) /= ((long double) StateFreq[i]);
	    }
	}

	for (int i = 0; i < StatesSize; i++)
	{
	    cout << "State " << i << ":" << endl;
	    for (int j = 0; j < LogOupSize; j++)
	    {
		cout << "Exp:\t" << ExpOupFreq(i,j) << "\tObs:\t" << ObsOupFreq(i,j) << endl;
	    }
	}

	for (int i = 0; i < LogOupSize; i++)
	{
	    cout << "Total amount of mark " << i << ": " << OupArray[i] << endl;
	}

	OupArray.resize(0);
	StateFreq.resize(0);
	ExpOupFreq.resize(0,0);
	ObsOupFreq.resize(0,0);
    }


    void ModelSpace::ShowResults(boost::multi_array<boost::numeric::ublas::vector<int>, 1>& ViterbiStates, boost::multi_array<boost::numeric::ublas::vector<int>, 1> & OupSerials, std::string OupFileName,int IndependentSerialsSize,boost::numeric::ublas::vector<unsigned long> &SerialsLengthArray)
    {
	ofstream OupFileStream;
	OupFileStream.open(OupFileName);
	if (OupFileStream.is_open())
	{

	    OupFileStream << "ID" << "\t" << "Index" << "\tObserved_Serial\t" << "\tHMM_state\t"<< endl;
	    for (int i = 0; i < IndependentSerialsSize; i++)
	    {
		std::string SerialName = std::to_string(i);
		unsigned long SerialLength = SerialsLengthArray[i];
		for (unsigned long j = 0; j < SerialLength; j++)
		{
		    OupFileStream << SerialName << "\t" << j << "\t" << OupSerials[i][j]<< "\t" <<ViterbiStates[i][j] << endl;
		}
	    }
	    OupFileStream.close();
	}
	else
	{
	    cerr << "Error opening an output file." << endl;
	}
    }

    void ModelSpace::ReadFromFile(const std::string &FileName,boost::multi_array<boost::numeric::ublas::vector<int>, 1> & ReadIn){
	std::ifstream NewFile;
	NewFile.open(FileName,std::ios::in);
	std::string tp;
	int flag=0;
	while(getline(NewFile,tp)&&NewFile.is_open())
	{
	    std::stringstream convert;
	    convert<<tp;

	    if(flag==0){
		std::string IndependentSerialsSize;
		std::string tmpSize;
		convert>>IndependentSerialsSize;
		int DataLength=std::stoi(IndependentSerialsSize);
		for(int itr=0;itr<DataLength;++itr)
		{
		    convert>>tmpSize;
		    ReadIn[itr].resize(std::stoi(tmpSize));
		    ReadIn[itr].clear();
		}
		flag=1;
		continue;
	    }

	    std::string LineString;
	    std::string pos1,pos2;
	    //while(getline(convert,LineString,'\t'))
	    {
		convert>>pos1;
		convert>>pos2;
		convert>>LineString;
		ReadIn[std::stoi(pos1)][std::stoi(pos2)]=std::stoi(LineString);
		//std::cout<<ReadIn[std::stoi(pos1)][std::stoi(pos2)]<<std::endl;
		//convert>>;
		//NewFile.close();
		//NewFile.close();
	    }
	}


    }


    template <typename T> void ModelSpace::ShowHMM(std::ostream& os,  boost::multi_array<std::unique_ptr<ModelStateTp<T>>, 1>&states, int StatesSize, int OupSize, boost::numeric::ublas::vector<T>& LeadingProbs)
    {

	os << StatesSize << " " << OupSize << std::endl;
	for (int i = 0; i < StatesSize; i++)
	{
	    for (int j = 0; j < StatesSize; j++)
	    {
		os << states[i]->TransValue(j) << " ";
	    }
	    os << std::endl;
	}
	for (int i = 0; i < StatesSize; i++)
	{
	    for (int j = 0; j < OupSize; j++)
	    {
		os << states[i]->EmitsValue(j) << " ";
	    }
	    os << std::endl;
	}
	for (int i = 0; i < StatesSize; i++)
	{
	    os << LeadingProbs[i] << " ";
	}
	os << std::endl;
    }


    class ScoreEstimator
    {
	public:
	    template<typename T> static LnDouble CalcScorePrH(int StatesSize, boost::multi_array<std::unique_ptr<ModelStateTp<T>>, 1>& states, boost::numeric::ublas::vector<T> &LeadingProb, boost::numeric::ublas::vector<int>& data, unsigned long SerialLength);

	private:

    };

    template<typename T> LnDouble ScoreEstimator::CalcScorePrH(int StatesSize, boost::multi_array<std::unique_ptr<ModelStateTp<T>>, 1>& states, boost::numeric::ublas::vector<T> &LeadingProb, boost::numeric::ublas::vector<int> &data, unsigned long SerialLength)
    {

	boost::numeric::ublas::vector<LnDouble> Alpha(StatesSize);
	Alpha.clear();
	boost::numeric::ublas::vector<LnDouble> tmpAlpha(StatesSize);
	tmpAlpha.clear();

	for (int i = 0; i < StatesSize; i++)
	{

	    Alpha[i] = LeadingProb[i] * states[i]->EmitsValue(data[0]);

	}
	for (unsigned long j = 1; j < SerialLength; j++)
	{
	    for (int k = 0; k < StatesSize; k++)
	    {
		tmpAlpha[k] = 0;
		for (int l = 0; l < StatesSize; l++)
		{

		    tmpAlpha[k] += Alpha[l] * states[l]->TransValue(k) * states[k]->EmitsValue(data[j]);

		}
	    }
	    Alpha.data().swap(tmpAlpha.data());
	}
	tmpAlpha.clear();
	tmpAlpha.resize(0);

	LnDouble prH=0;
	for (int j = 0; j < StatesSize; j++)
	{
	    prH += Alpha[j];
	}
	Alpha.resize(0);


	return prH;
    }


    class ModelManager
    {
	public:
	    static double  CalcScoreBIC(int StatesSize, int OupSize, double LikelihoodScore, unsigned long SerialsTotalLength);
	    static void RandomModelInit(int StatesSize, int OupSize, variate_generator<mt19937, uniform_real<> >& rand, boost::numeric::ublas::vector<double>& LeadingProb);
	    static void ModelCleaner(int StatesSize, boost::multi_array<std::unique_ptr<ModelStateTp<double>>, 1>& HMM);
    };


    class BaumWelchScaler
    {
	public:

	    template <typename T> static void BWScale(int OupSize, int StatesSize, boost::multi_array<std::unique_ptr<ModelStateTp<T>>, 1>&states,boost::numeric::ublas::vector<T>& LeadingProb, boost::numeric::ublas::vector<int>& data, unsigned long SerialLength,boost::numeric::ublas::matrix<T> & ArrayOfGamma,boost::numeric::ublas::matrix<T>&DeltaMatrix,boost::multi_array<T, 3> &TransEmitsTensor);
    };

    template <typename T> void BaumWelchScaler::BWScale(int OupSize, int StatesSize,boost::multi_array<std::unique_ptr<ModelStateTp<T>>, 1>&states,  boost::numeric::ublas::vector<T>& LeadingProb, boost::numeric::ublas::vector<int>& data, unsigned long SerialLength, boost::numeric::ublas::matrix<T>& GammaSumMatrix, boost::numeric::ublas::matrix<T>&DeltaMatrix,boost::multi_array<T, 3> &TransEmitsTensor)
    {

	Eigen::Array<T,1,Eigen::Dynamic> ScaleFactors;
	ScaleFactors.resize(SerialLength);

	boost::numeric::ublas::vector<T> OldBeta;
	OldBeta.resize(StatesSize);
	OldBeta.clear();

	boost::numeric::ublas::vector<T> NewBeta;
	NewBeta.resize(StatesSize);
	NewBeta.clear();

	boost::numeric::ublas::matrix<T> TmpMatrix;
	TmpMatrix.resize(StatesSize,SerialLength);
	TmpMatrix.clear();

	GammaSumMatrix.clear();
	ScaleFactors(0)=0;
	for (int i = 0; i < StatesSize; i++)
	{
	    TmpMatrix(i,0) = LeadingProb[i] * states[i]->EmitsValue(data[0]);
	    ScaleFactors(0) += TmpMatrix(i,0);
	}
	for (int i = 0; i < StatesSize; i++)
	{
	    TmpMatrix(i,0) /= ScaleFactors(0);
	}

	for (unsigned long j = 1; j < SerialLength; j++)
	{
	    ScaleFactors(j) = 0;
            #pragma omp parallel
	    {
                #pragma omp for
		for (int s = 0; s < StatesSize; s++)
		{
		    T tmpVal = 0;
		    for (int t = 0; t < StatesSize; t++)
		    {
			tmpVal += TmpMatrix(t,j-1) * TransEmitsTensor[data[j]][t][s];
		    }
		    TmpMatrix(s,j) = tmpVal;
		    ScaleFactors(j) += tmpVal;
		}
                #pragma omp for
		for (int s = 0; s < StatesSize; s++)
		{
		    TmpMatrix(s,j) /= ScaleFactors(j);
		}
	    }
	}
	//PrH
	//j + 1 used because it is unsigned, and checking >= 0 would always be true!
	DeltaMatrix.clear();
	for (int t = 0; t < StatesSize; t++)
	{
	    OldBeta(t) = 1.0 / ScaleFactors(SerialLength - 1);
	    for (int s = 0; s < StatesSize; s++)
	    {
		GammaSumMatrix(s,t) += OldBeta(t)*TransEmitsTensor[data[SerialLength - 1]][s][t]*TmpMatrix(s,SerialLength - 2);
		DeltaMatrix(s,SerialLength - 2) += OldBeta(t)*TransEmitsTensor[data[SerialLength - 1]][s][t]*TmpMatrix(s,SerialLength - 2);
	    }
	}
	for (unsigned long JReverse = SerialLength - 1; JReverse > 1; JReverse--)
	{
            #pragma omp parallel for
	    for (int t = 0; t < StatesSize; t++)
	    {
		T tmpVal = 0;
		for (int u = 0; u < StatesSize; u++)
		{
		    tmpVal += TransEmitsTensor[data[JReverse]][t][u] * OldBeta(u);
		}
		NewBeta(t)=tmpVal / ScaleFactors(JReverse - 1);
		for (int s = 0; s < StatesSize; s++)
		{
		    GammaSumMatrix(s,t) += NewBeta(t)*TransEmitsTensor[data[JReverse - 1]][s][t]*TmpMatrix(s,JReverse - 2);
		    DeltaMatrix(s,JReverse - 2)+=NewBeta(t)*TransEmitsTensor[data[JReverse - 1]][s][t]*TmpMatrix(s,JReverse - 2);
		}
	    }
	    OldBeta.data().swap(NewBeta.data());
	}


	//DeltaArray
	for (int s = 0; s < StatesSize; s++)
	{
	    DeltaMatrix(s,SerialLength - 1)= TmpMatrix(s,SerialLength - 1);//prH;
	}
	TmpMatrix.resize(0,0);
	ScaleFactors.resize(0);
    }

    class GPU
    {
	public:
	    template <typename T> T CudaVectorSum(std::vector<T> & vet);
    };

    template <typename T> __global__ void VectorSumKernel(T *d_in, T *d_out, int n);
    template <typename T> __global__ void VectorSumKernel(T *d_in, T *d_out, int n)
    {
	__shared__ T sharedData[BLOCK_SIZE];  // Shared memory for the block
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// Initialize shared memory to 0
	if (index < n) {
	    sharedData[threadIdx.x] = d_in[index];
	} else {
	    sharedData[threadIdx.x] = 0;
	}

	__syncthreads();

	// Perform reduction within the block
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) 
	{
	    if (threadIdx.x < stride) 
	    {
		sharedData[threadIdx.x] += sharedData[threadIdx.x + stride];
	    }
	    __syncthreads();
	}

	// Write the block result to global memory
	if (threadIdx.x == 0) 
	{
	    d_out[blockIdx.x] = sharedData[0];
	}
    }
    template <typename T> T CudaVectorSum(std::vector<T> & vet)
    {

	int length=vet.size();

	T *h_out;
	T *d_in, *d_out;
	T SumResult = 0;

	h_out = (T*)calloc( (length / BLOCK_SIZE + 1),sizeof(T));

	// Allocate memory on the device
	cudaMalloc(&d_in, length * sizeof(T));
	cudaMalloc(&d_out, sizeof(T) * (length / BLOCK_SIZE + 1));
	// Copy input data from host to device

	cudaMemcpy(d_in, vet.data(), length * sizeof(T), cudaMemcpyHostToDevice);

	// Launch the kernel with enough blocks to process the vector
	int blocks = (length + BLOCK_SIZE) / BLOCK_SIZE;
	VectorSumKernel<T><<<blocks, BLOCK_SIZE>>>(d_in, d_out, length);
	// Copy the result from device to host
	cudaMemcpy(h_out, d_out, sizeof(T) * blocks, cudaMemcpyDeviceToHost);

	// Final reduction on the host
	for (int i = 0; i < blocks; i++) 
	{
	    SumResult += h_out[i];
	}

	free(h_out);
	cudaFree(d_in);
	cudaFree(d_out);

	return SumResult;
    }



    class ModelRenewer
    {
	public:
	    template <typename T> static void UpdateModelParams(boost::multi_array<std::unique_ptr<ModelStateTp<T>>, 1> &ModelObj,int StatesSize, int OupSize, int IndependentSerialsSize, boost::multi_array<boost::numeric::ublas::matrix<T>, 1>& DeltaArray, boost::multi_array<boost::numeric::ublas::matrix<T>, 1> &GammaSumMatrix, boost::multi_array<boost::numeric::ublas::vector<int>, 1>& data, boost::numeric::ublas::vector<unsigned long>& SerialLengthArray, boost::numeric::ublas::vector<T>& LeadingProb);
    };



    template <typename T> void ModelRenewer::UpdateModelParams(boost::multi_array<std::unique_ptr<ModelStateTp<T>>, 1> &ModelObj,int StatesSize, int OupSize, int IndependentSerialsSize, boost::multi_array<boost::numeric::ublas::matrix<T>, 1>&DeltaArray, boost::multi_array<boost::numeric::ublas::matrix<T>, 1> &GammaSumMatrix, boost::multi_array<boost::numeric::ublas::vector<int>, 1>& data, boost::numeric::ublas::vector<unsigned long>& SerialLengthArray, boost::numeric::ublas::vector<T>& LeadingProb)
    {
	//start probs
	if (LeadingProb.data().begin() != NULL)
	{
	    T Denom = 0;
	    {
		for (int i = 0; i < StatesSize; i++)
		{

		    std::vector<T> slice_elements;
		    slice_elements.reserve(DeltaArray.size()); 
		    std::transform(DeltaArray.begin(), DeltaArray.end(), std::back_inserter(slice_elements),
			    [&](const boost::numeric::ublas::matrix<T>& mat) { return mat(i, 0); });
		    Denom+=CudaVectorSum<T>(slice_elements);
		}
		for (int i = 0; i < StatesSize; i++)
		{
		    LeadingProb[i] = 0;

		    std::vector<T> slice_elements;
		    slice_elements.reserve(DeltaArray.size()); 
		    std::transform(DeltaArray.begin(), DeltaArray.end(), std::back_inserter(slice_elements),
			    [&](const boost::numeric::ublas::matrix<T>& mat) { return mat(i, 0); });
		    LeadingProb[i]+=CudaVectorSum<T>(slice_elements);
		    LeadingProb[i] /= Denom;
		}
	    }
	}
	//trans probs
	{
	    boost::multi_array<T, 2> GammaDenomMatrix(boost::extents[StatesSize][IndependentSerialsSize]);
            #pragma omp parallel
	    {
                #pragma omp for
		for (int i = 0; i < IndependentSerialsSize; i++)
		{
		    for (int j = 0; j < StatesSize; j++)
		    {
			GammaDenomMatrix[j][i] = 0;
			for (int k = 0; k < StatesSize; k++)
			{
			    GammaDenomMatrix[j][i] += GammaSumMatrix[i](j,k);
			}
		    }
		}
                #pragma omp for
		for (int i = 0; i < StatesSize; i++)
		{
		    for (int j = 0; j < StatesSize; j++)
		    {
			T SerialSum = 0;
			T SerialSumDenom = 0;

			std::vector<T> slice_elem1;
			slice_elem1.reserve(GammaSumMatrix.size()); 
			std::transform(GammaSumMatrix.begin(), GammaSumMatrix.end(), std::back_inserter(slice_elem1),
				[&](const boost::numeric::ublas::matrix<T>& mat) { return mat(i, j); });

			SerialSum+=CudaVectorSum<T>(slice_elem1);

			std::vector<T> slice_elem2(GammaDenomMatrix[i].begin(),GammaDenomMatrix[i].end());
			SerialSumDenom+=CudaVectorSum<T>(slice_elem2);

			ModelObj[i]->SetTransValue(j, SerialSum / SerialSumDenom);
		    }
		}
	    }

	    GammaDenomMatrix.resize(boost::extents[0][0]);
	}
	//emit probs
	{
	    boost::multi_array<T, 1> SumDelta(boost::extents[StatesSize]);
	    std::fill_n(SumDelta.data(), SumDelta.num_elements(), 0);

	    for (int i = 0; i < StatesSize; i++)
	    {
		SumDelta[i] = 0;
		for (int seq = 0; seq < IndependentSerialsSize; seq++)
		{
		    auto rowMatrix = boost::numeric::ublas::row(DeltaArray[seq], i);
		    std::vector<T> slice_Mat(rowMatrix.begin(),rowMatrix.end());
		    SumDelta[i]+=CudaVectorSum<T>(slice_Mat);

		}
	    }
            #pragma omp parallel
	    {
                #pragma omp for schedule(dynamic)
		for (int i = 0; i < StatesSize; i++)
		{
		    for (int k = 0; k < OupSize; k++)
		    {
			T tmpSum = 0;
			for (int seq = 0; seq < IndependentSerialsSize; seq++)
			{
			    for (unsigned long j = 0; j < SerialLengthArray[seq]; j++)
			    {
				if (k == data[seq][j]) tmpSum += DeltaArray[seq](i,j);
			    }
			}
			ModelObj[i]->SetEmitsValue(k, tmpSum / SumDelta[i]);
		    }
		}
	    }

	    SumDelta.resize(boost::extents[0]);
	}
    }


    class ViterbiDecoder
    {
	public:
	    template<typename T> static void viterbi(int StatesSize, int OupSize, boost::numeric::ublas::vector<int> &Viterbi_states,boost::multi_array<std::unique_ptr<ModelStateTp<T>>, 1> &states , boost::numeric::ublas::vector<T>& LeadingProb,  boost::numeric::ublas::vector<int> &data, unsigned long SerialLength);

    };

    template<typename T> void ViterbiDecoder::viterbi(int StatesSize, int OupSize, boost::numeric::ublas::vector<int> &Viterbi_states,boost::multi_array<std::unique_ptr<ModelStateTp<T>>, 1> &states , boost::numeric::ublas::vector<T>& LeadingProb,  boost::numeric::ublas::vector<int> &data, unsigned long SerialLength)
    {
	Viterbi_states.resize(SerialLength);
	Viterbi_states.clear();

	boost::numeric::ublas::matrix<LnDouble> Viterbi_delta;
	boost::numeric::ublas::matrix<int> Viterbi_psi;

	Viterbi_delta.resize(StatesSize,SerialLength);
	Viterbi_delta.clear();

	Viterbi_psi.resize(StatesSize,SerialLength);
	Viterbi_psi.clear();
	for (int i = 0; i < StatesSize; i++)
	{
	    Viterbi_delta(i,0)=LeadingProb[i] * states[i]->EmitsValue(data[0]);
	}
	//forward sweep
	for (unsigned long t = 1; t < SerialLength; t++)
	{
	    for (int j = 0; j < StatesSize; j++)
	    {
		LnDouble tmpMax = Viterbi_delta(0,t-1) * states[0]->TransValue(j);
		int IndexFlagForMax = 0;
		for (int i = 1; i < StatesSize; i++)
		{
		    LnDouble tmpVal = Viterbi_delta(i,t-1) * states[i]->TransValue(j);
		    if (tmpVal > tmpMax)
		    {
			tmpMax = tmpVal;
			IndexFlagForMax = i;
		    }
		}
		Viterbi_delta(j,t) = tmpMax * states[j]->EmitsValue(data[t]);
		Viterbi_psi(j,t) = IndexFlagForMax;
	    }
	}
	//get q(tMax)
	LnDouble tmpMax = Viterbi_delta(0,SerialLength-1);
	int IndexFlagForMax = 0;
	for (int i = 1; i < StatesSize; i++)
	{
	    if (Viterbi_delta(i,SerialLength-1) > tmpMax)
	    {
		tmpMax = Viterbi_delta(i,SerialLength-1);
		IndexFlagForMax = i;
	    }
	}
	Viterbi_states[SerialLength - 1] = IndexFlagForMax;
	for (unsigned long t = SerialLength - 1; t > 0; --t) //must use --t since it is unsigned
	{
	    Viterbi_states[t-1] = Viterbi_psi(Viterbi_states[t],t);
	}

	Viterbi_delta.resize(0,0);
	Viterbi_psi.resize(0,0);

    }


    int ModelSpace::learn(int repsFlag)
    {

	int IndependentSerialsSize = 20;
	int StatesSize = 5;
	int OupSize = 6;
	int SeqLengthForTesting = 5000;
	//timeval curTime;
	//gettimeofday(&curTime, NULL);
	//mt19937 gen( static_cast<unsigned int>(curTime.tv_usec));
	mt19937 gen( static_cast<unsigned int>(10));
	variate_generator<mt19937, uniform_real<> > rand(gen, uniform_real<double>(0.0, 1.0));
	//std::srand(time(NULL));
	std::srand(10);

	////////////////////////GET DATA////////////////////////////////////////
	boost::multi_array<boost::numeric::ublas::vector<int>, 1> EventsSerial(boost::extents[IndependentSerialsSize]);
	boost::numeric::ublas::vector<unsigned long> SerialsLengthArray;

	unsigned long SerialsTotalLength = 0;

	bool GetDataFromFile=true; //false
	if(GetDataFromFile){
	    ReadFromFile("./SerialsToLearn.txt",EventsSerial);
	    SerialsLengthArray.resize(EventsSerial.shape()[0]);
	    SerialsLengthArray.clear();
	    for (int i = 0; i < EventsSerial.shape()[0]; i++)
	    {
		SerialsLengthArray[i]=EventsSerial[i].size();
		SerialsTotalLength += EventsSerial[i].size();
	    }
	}else{
	    SerialsLengthArray.resize(EventsSerial.shape()[0]);
	    SerialsLengthArray.clear();
	    for (int i = 0; i < IndependentSerialsSize; i++)
	    {
		SerialsLengthArray[i] = SeqLengthForTesting + 1000 * i;
		SerialsTotalLength += SerialsLengthArray[i];
	    }
	    for (int seq = 0; seq < IndependentSerialsSize; seq++)
	    {
		EventsSerial[seq].resize(SerialsLengthArray[seq]);
		EventsSerial[seq].clear();
		for (unsigned int i = 0; i < SerialsLengthArray[seq]; i++)
		{
		    //add an artificial bias so there's actually something to learn besides a flat distribution
		    if (std::rand() % 3 == 0) EventsSerial[seq][i] = 2;
		    else EventsSerial[seq][i] = (std::rand() % OupSize);
		}
	    }

	    std::ofstream OupStream("./SerialsToLearn_PrintedOut.txt");
	    if (!OupStream) {
		std::cerr << "Error opening the file!" << std::endl;
		return 1;
	    }
	    OupStream<<IndependentSerialsSize;
	    for (int seq = 0; seq < IndependentSerialsSize; seq++){
		OupStream<<'\t'<<SerialsLengthArray[seq];
	    }
	    OupStream<<endl;
	    for (int seq = 0; seq < IndependentSerialsSize; seq++)
	    {
		for (unsigned int i = 0; i < SerialsLengthArray[seq]; i++)
		{
		    OupStream<<seq<<'\t'<<i<<'\t'<<EventsSerial[seq][i]<<endl;
		}
	    }
	}
	/////////////////////////////////////////////////////////////////////


	//////////////////////CREATE MODEL///////////////////////////////////////////////
	boost::multi_array<std::unique_ptr<ModelStateTp<double>>, 1> ModelObj(boost::extents[StatesSize]);
	int count=1;
	for (auto &hmm : ModelObj)
	{
	    cout << "HMM " << count << ":" << endl;
	    count++;
	    hmm=std::make_unique<ModelStateTp<double>>();
	    hmm->FillModelState(StatesSize,OupSize);
	    hmm->FillRandom();
	}

	boost::numeric::ublas::vector<double> LeadingProbs;
	LeadingProbs.resize(StatesSize);
	LeadingProbs.clear();
	ModelManager::RandomModelInit(StatesSize, OupSize, rand,  LeadingProbs);

	double LikelihoodScore = 0;
	for (int i = 0; i < IndependentSerialsSize; i++)
	{
	    //likelihood under the PRIOR HMM
	    LikelihoodScore += ScoreEstimator::CalcScorePrH<double>(StatesSize, ModelObj, LeadingProbs, EventsSerial[i], SerialsLengthArray[i]).getLog();
	}
	double SecondBIC = ModelManager::CalcScoreBIC(StatesSize, OupSize, LikelihoodScore, SerialsTotalLength);
	cout << "Initial BIC: " << SecondBIC << endl;
	cout << endl;
	//////////////////////CREATE MODEL END///////////////////////////////////////////////


	//////////////////////CREATE WORKING SPACE///////////
	boost::multi_array<boost::numeric::ublas::matrix<double>, 1> ArrayOfDelta(boost::extents[IndependentSerialsSize]);
	boost::multi_array<boost::numeric::ublas::matrix<double>, 1> ArrayOfGamma(boost::extents[IndependentSerialsSize]);
	for (int seq = 0; seq < IndependentSerialsSize; seq++)
	{
	    ArrayOfDelta[seq].resize(StatesSize,SerialsLengthArray[seq]);
	    ArrayOfDelta[seq].clear();
	    ArrayOfGamma[seq].resize(StatesSize,StatesSize);
	    ArrayOfGamma[seq].clear();
	}

	boost::multi_array<double, 3> TransEmitsTensor(boost::extents[OupSize][StatesSize][StatesSize]);
	std::fill_n(TransEmitsTensor.data(), TransEmitsTensor.num_elements(), 0);
	/////////////////////////////////
	for (int iteration = 0; iteration < std::max(1,repsFlag) ; iteration+= (repsFlag != -1 ? 1 : 0))
	{
	    //Update TransEmitsTensor
	    for (int k = 0; k < OupSize; k++)
	    {
		for (int i = 0; i < StatesSize; i++)
		{
		    for (int j = 0; j < StatesSize; j++)
		    {
			TransEmitsTensor[k][i][j] = ModelObj[i]->TransValue(j) * ModelObj[j]->EmitsValue(k);
		    }
		}
	    }


	    for (int i = 0; i < IndependentSerialsSize; i++)
	    {
		BaumWelchScaler::BWScale<double>(OupSize, StatesSize, ModelObj, LeadingProbs, EventsSerial[i], SerialsLengthArray[i], ArrayOfGamma[i], ArrayOfDelta[i],TransEmitsTensor);
	    }

	    ModelRenewer::UpdateModelParams<double>(ModelObj,StatesSize, OupSize, IndependentSerialsSize, ArrayOfDelta, ArrayOfGamma, EventsSerial, SerialsLengthArray, LeadingProbs);

	    cout << "Iteration " << iteration + 1 << ":" << endl;
	    double LikelihoodScore = 0;
	    for (int i = 0; i < IndependentSerialsSize; i++)
	    {
		//likelihood under the PRIOR HMM
		LikelihoodScore += ScoreEstimator::CalcScorePrH<double>(StatesSize, ModelObj, LeadingProbs, EventsSerial[i], SerialsLengthArray[i]).getLog();
	    }
	    double FirstBIC = ModelManager::CalcScoreBIC(StatesSize, OupSize, LikelihoodScore, SerialsTotalLength);
	    cout << "BIC: " << FirstBIC << endl;
	    cout<<"Delta="<<(SecondBIC - FirstBIC)<< endl;
	    cout<<endl;

	    if (SecondBIC - FirstBIC < 0)
	    {
		std::cerr << "ERROR: BIC increasing" << std::endl;
	    }
	    else if (SecondBIC - FirstBIC < 10 )
	    {
		//change in BIC below threshold, stop training
		repsFlag = 0;
	    }
	    SecondBIC = FirstBIC;
	}

	ofstream HMMstream;
	HMMstream.open("./HMMmodel.txt");
	ShowHMM<double>(HMMstream,ModelObj,StatesSize,OupSize, LeadingProbs);
	HMMstream.close();


	boost::multi_array<boost::numeric::ublas::vector<int>, 1> ViterbiStates(boost::extents[IndependentSerialsSize]);
	cout << "Calling states....." << endl;
	for (int i = 0; i < IndependentSerialsSize; i++)
	{
	    ViterbiDecoder::viterbi<double>(StatesSize, OupSize,ViterbiStates[i], ModelObj, LeadingProbs, EventsSerial[i], SerialsLengthArray[i]);
	}
	cout << "Done." << endl;

	cout << "Writing results....." << endl;
	ShowResults(ViterbiStates, EventsSerial, "SerialsAndStates.txt",IndependentSerialsSize,SerialsLengthArray);
	cout << "Done." << endl;


	ShowMarkStates(IndependentSerialsSize,OupSize,StatesSize,ModelObj, SerialsLengthArray, EventsSerial,ViterbiStates);
	ShowMarkStatesPerObservation(IndependentSerialsSize,OupSize,StatesSize,ModelObj, SerialsLengthArray, EventsSerial,ViterbiStates);



	///////////free memory///////////
	TransEmitsTensor.resize(boost::extents[0][0][0]);
	ModelManager::ModelCleaner(StatesSize, ModelObj);
	for (auto &delta : ArrayOfDelta)
	{
	    delta.resize(0,0);
	}
	ArrayOfDelta.resize(boost::extents[0]);
	for (auto &gamma : ArrayOfGamma)
	{
	    gamma.resize(0,0);
	}
	ArrayOfGamma.resize(boost::extents[0]);

	for (auto &foo : EventsSerial)
	{
	    foo.resize(0);
	}
	EventsSerial.resize(boost::extents[0]);

	for (auto &foo : ViterbiStates)
	{
	    foo.resize(0);
	}
	ViterbiStates.resize(boost::extents[0]);

	SerialsLengthArray.resize(0);
	LeadingProbs.resize(0);

	return EXIT_SUCCESS;
    }


    double  ModelManager::CalcScoreBIC(int StatesSize, int OupSize, double LikelihoodScore, unsigned long SerialsTotalLength)
    {
	return (double) -2.0 * LikelihoodScore + ((((double) StatesSize) * (((double) StatesSize - 1.0) + ((double) OupSize - 1.0)) + ((double) StatesSize)) * log((double) SerialsTotalLength));
    }


    void ModelManager::RandomModelInit(int StatesSize, int OupSize, variate_generator<mt19937, uniform_real<> >& rand, boost::numeric::ublas::vector<double>& LeadingProb)
    {
	double count = 0;
	//avoids a potential bug: what if the variate_generator passed to this was initialized at the same second as the static variate_generator held by HMM::State?
	for (int j = 0; j < (StatesSize * OupSize) + (StatesSize * StatesSize); j++)
	{
	    rand();
	}
	for (int j = 0; j < StatesSize; j++)
	{
	    LeadingProb[j] = rand();
	    count += LeadingProb[j];
	}
	for (int j = 0; j < StatesSize; j++)
	{
	    LeadingProb[j] /= count;
	}
    }

    void ModelManager::ModelCleaner(int StatesSize, boost::multi_array<std::unique_ptr<ModelStateTp<double>>, 1>& HMM)
    {
	for(auto &hmm:HMM){
	    hmm.reset();
	}
    }



}

