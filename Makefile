CXX = nvcc -std=c++17 -O3
FLAG = --threads 4 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_75,code=sm_75
OUT = code
OBJDIR = ./objects
SRCDIR = ./src


LIBOBJS= \
	$(OBJDIR)/HiddenMarkovModel.o 


$(OUT): objdir $(LIBOBJS)
	$(CXX) $(FLAG) $(LIBOBJS) -o $@


objdir:
	mkdir -p $(OBJDIR)

$(OBJDIR)/HiddenMarkovModel.o: $(SRCDIR)/HiddenMarkovModel.cu
	$(CXX) $(FLAG) -g -c $< -o $@


clean:
	rm -rf $(OBJDIR) $(OUT)
