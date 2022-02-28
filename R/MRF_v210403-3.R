
MRF = function(data,x.pos,oos.pos, y.pos=1, S.pos=2:ncol(data),
               minsize=10,mtry.frac=1/3,min.leaf.frac.of.x=1,VI=FALSE,
               ERT=FALSE,quantile.rate=NULL,S.priority.vec=NULL,
               random.x = FALSE,howmany.random.x=1,howmany.keep.best.VI=20,
               cheap.look.at.GTVPs=TRUE,
               prior.var=c(),prior.mean=c(),
               subsampling.rate=0.75,rw.regul=0.75,keep.forest=FALSE,
               block.size=12, trend.pos=max(S.pos),trend.push=1, fast.rw=TRUE,
               ridge.lambda=0.1,HRW=0,B=50,resampling.opt=2,printb=TRUE){
  
  ######## TRANSLATION BLOCK + MISC ##################
  rownames(data)=c()
  ET=ERT #translation to older names in the code
  ET.rate = quantile.rate #translation to older names in the code
  z.pos = x.pos #translation to older names in the code
  x.pos= S.pos #translation to older names in the code
  random.z= random.x #translation to older names in the code
  min.leaf.fracz= min.leaf.frac.of.x #translation to older names in the code
  random.z.number= howmany.random.x #translation to older names in the code
  x.pos= S.pos #translation to older names in the code
  VI.rep=10*VI #translation to older names in the code
  bootstrap.opt=resampling.opt #translation to older names in the code
  regul.lambda = ridge.lambda   #translation to older names in the code
  prob.vec = S.priority.vec #translation to older names in the code
  keep.VI= howmany.keep.best.VI  #translation to older names in the code
  no.rw.trespassing=TRUE #used to be an option, but turns out I can't think of a good reason to choose 'FALSE'. So I impose it.
  BS4.frac=subsampling.rate #translation to older names in the code
  trend.pos = which(S.pos==trend.pos) #translate trend.pos to be interms of position in S
  if(!is.null(prior.var)){prior.var = 1/prior.var} #those are infact implemented in terms of heterogeneous lambdas
  ###################################################
  
  ######## FOOLPROOF ################################
  if(length(z.pos)>10 & (regul.lambda<0.25 | rw.regul==0)){stop('For models of this size, consider using a higher ridge lambda (>0.25) and RW regularization.')}
  if(min(x.pos)<1){stop('S.pos cannot be non-postive.')}
  if(max(x.pos)>ncol(data)){stop('S.pos specifies variables beyond the limit of your data matrix.')}
  if(is.element(y.pos,c(x.pos))){x.pos=x.pos[x.pos!=y.pos]
  print('Beware: foolproof 1 activated. S.pos and y.pos overlap.')
  trend.pos=which(x.pos==trend.pos)
  } #foolproof 1
  if(is.element(y.pos,c(z.pos))){z.pos=z.pos[z.pos!=y.pos]
  print('Beware: foolproof 2 activated. x.pos and y.pos overlap.')} #foolproof 2
  if(regul.lambda<0.0001){regul.lamba=0.0001
  print('Ridge lambda was too low or negative. Was imposed to be 0.0001')} #foolproof 3
  if(!is.null(ET.rate)){if(ET.rate<0){ET.rate=0
  print('Quantile Rate/ERT rate was forced to 0, It cannot be negative.')}} #foolproof 4
  if(!is.null(prior.mean) & random.z==TRUE){print('Are you sure you want to mix a customized prior with random X?')}
  if(length(z.pos)<1){stop('You need to specificy at least one X.')}
  if(length(prior.var)!=0 | length(prior.mean)!=0){
    if(is.null(prior.var) & !is.null(prior.mean)){stop('Need to additionally specificy prior.var.')}
    if(is.null(prior.mean) & !is.null(prior.var)){stop('Need to additionally specificy prior.mean.')}
    if(length(prior.mean)!=(length(z.pos)+1) | length(prior.var)!=(length(z.pos)+1)){stop('Length of prior vectors do not match that of X.')}}
  if(length(x.pos)<5){print('Your S_t is small. Consider augmentating with noisy carbon copies it for better results.')}
  if((length(x.pos)*mtry.frac)<1){stop('Mtry.frac is too small.')}
  if(min.leaf.fracz>2){min.leaf.fracz=2
  print('min.leaf.frac.of.x forced to 2. Let your trees run deep!')}
  if(is.null(colnames(data))){
    stop('Data frame must have column names.')}
  if((min.leaf.fracz*(length(z.pos)+1)) < 2){min.leaf.fracz=2/(length(z.pos)+1)
  print(paste('Min.leaf.frac.of.x was too low. Thus, it was forced to ',2/(length(z.pos)+1),' -- a bare minimum. You should consider a higher one.',sep=''))} #foolproof 3
  ###################################################
  
  ########## Cheap trick to avoids matrix bigs when oos.pos is void
  if(length(oos.pos)==0){
    oos.flag=TRUE
    shit=matrix(0,2,ncol(data))
    colnames(shit)=colnames(data)
    new.data = rbind(data,shit)
    original.data = data
    #original.oos.pos = oos.pos
    data = new.data
    oos.pos = (nrow(data)-1):nrow(data)
    fake.pos = oos.pos
    rownames(data)=c()
  }else{oos.flag=FALSE}
  
  ######## SETUP ARRAYS AND STUFF ###################
  dat=data
  K=length(z.pos)+1
  if(random.z==TRUE){K=random.z.number+1}
  
  commitee = matrix(NA,B,length(oos.pos))
  avg.pred = rep(0,length(oos.pos))
  pred.kf = array(NA,dim=c(B,nrow(data),length(x.pos)+1))
  #pred.kf.group = array(NA,dim=c(B,nrow(data),length(unique(g.vec))))
  all.fits = matrix(NA,B,nrow(data)-length(oos.pos))
  
  avg.beta = matrix(0,nrow(data),K)
  whos.in.mat = matrix(0,nrow(data),B)
  betas.draws =  array(0,dim=c(dim(avg.beta),B))
  betas.draws.nonOVF = betas.draws
  #if(random.z==TRUE){avg.beta=matrix(0,nrow(data),random.z.number+1)} #override
  
  betas.shu = array(0,dim=c(dim(avg.beta),length(x.pos)+1))
  betas.shu.nonOVF = array(0,dim=c(dim(avg.beta),length(x.pos)+1))
  #betas.shu.group = array(0,dim=c(dim(avg.beta),length(unique(g.vec))))
  avg.beta.nonOVF=avg.beta
  forest=list()
  random.vecs=list()
  ####################################################
  
  ###################################################################################
  ###################################################################################
  ########################## Ensemble Loop ##########################################
  ###################################################################################
  ###################################################################################
  
  for(b in 1:B){
    if(printb==TRUE){print(paste('Tree ',b,' out of ',B,sep=''))}
    
    ############################################
    ###### Pick your resampling scheme #########
    ############################################
    if(bootstrap.opt==0){ #No Bootstrap/sub-sampling. Should not be used for looking at GTVPs.
      chosen.ones.plus = 1:(nrow(dat[-oos.pos,]))
      rando.vec = c(sort(chosen.ones.plus))}
    if(bootstrap.opt==1){ #plain sub-sampling
      chosen.ones = sample(x=1:(nrow(dat[-oos.pos,])),replace = FALSE,size=BS4.frac*nrow(dat[-oos.pos,]))
      chosen.ones.plus = c(chosen.ones)
      rando.vec = c(sort(chosen.ones.plus))
    }
    if(bootstrap.opt==2){ #block sub-sampling. recommended when looking at TVPs.
      groups=sort(base::sample(x=c(1:(nrow(dat[-oos.pos,])/block.size)),size=nrow(dat[-oos.pos,]),replace=TRUE))
      rando.vec = rexp(rate=1,n=nrow(dat[-oos.pos,])/block.size)[groups] +0.1
      chosen.ones.plus=rando.vec
      rando.vec=which(chosen.ones.plus>quantile(chosen.ones.plus,1-BS4.frac))
      chosen.ones.plus=rando.vec
      rando.vec = c(sort(chosen.ones.plus))
    }
    if(bootstrap.opt>=3){ #plain bayesian bootstrap
      chosen.ones = rexp(rate=1,n=nrow(dat[-oos.pos,]))+0.1
      chosen.ones.plus = chosen.ones/mean(chosen.ones) #sum, for dirichlet
      rando.vec = chosen.ones.plus
      
      if(bootstrap.opt==4){ #Block Bayesian Bootstrap. Recommended for forecasting.
        groups=sort(base::sample(x=c(1:(nrow(dat[-oos.pos,])/block.size)),size=nrow(dat[-oos.pos,]),replace=TRUE))
        rando.vec = rexp(rate=1,n=nrow(dat[-oos.pos,])/block.size)[groups] +0.1
        chosen.ones.plus=rando.vec
        chosen.ones.plus = chosen.ones.plus/mean(chosen.ones.plus) #sum, for dirichlet
        rando.vec = chosen.ones.plus
      }
    }
    #############################################
    #############################################
    
    #Should we randomize z? Only briefly used in the appendix of the paper for EOTB-MRF.
    # This cannot be used if one wants to look at GTVPs.
    if(random.z==TRUE){z.pos.effective = sample(x=z.pos,size=random.z.number,replace=FALSE)}
    else{z.pos.effective=z.pos}
    
    rt.output=one.mrf.tree(data=dat,minsize=minsize,min.leaf.fracz=min.leaf.fracz,rw.regul=rw.regul,VI.rep=VI.rep,
                           #g.vec=g.vec,balancing=balancing,
                           trend.push=trend.push,trend.pos=trend.pos,y.pos=y.pos,
                           #Hmax=Hmax,it.forecast = it.forecast,
                           prior.var=prior.var,prior.mean=prior.mean,prob.vec = prob.vec,
                           rando.vec=rando.vec,ET=ET,ET.rate=ET.rate,
                           #skip.VI.individual=skip.VI.individual,
                           z.pos = z.pos.effective,x.pos=x.pos,oos.pos=oos.pos,
                           frac=mtry.frac,HRW=HRW, regul.lambda=regul.lambda,
                           no.rw.trespassing = no.rw.trespassing,fast.rw=fast.rw)
    
    if(keep.forest){
      forest[[b]]=rt.output$tree
      random.vecs[[b]]=rando.vec
    }
    
    if(bootstrap.opt==3 |bootstrap.opt==4){ #for Bayesian Bootstrap, gotta impose a cutoff on what is OOS and what is not.
      rando.vec=which(chosen.ones.plus>quantile(chosen.ones.plus,.5))
      rt.output$betas[is.na(rt.output$betas)]=0
      rt.output$pred[is.na(rt.output$pred)]=0
    }
    
    #stack the trees prediction and update the mean prediction
    commitee[b,]=rt.output$pred
    avg.pred = ((b-1)/b)*avg.pred +  (1/b)*(rt.output$pred)
    
    #accounting which observtaions were used or not used to contruct this particular tree
    in.out = rep(0,nrow(data))
    in.out[-rando.vec]=1
    whos.in.mat[,b]=in.out
    
    #stack the betas and update average betas.
    avg.beta[,] = ((b-1)/b)*avg.beta + (1/b)*rt.output$betas
    betas.draws[,,b] = rt.output$betas
    
    rt.output$betas[in.out==0,]=rep(0,length(z.pos.effective)+1)
    avg.beta.nonOVF = avg.beta.nonOVF + rt.output$betas
    rt.output$betas[in.out==0,]=rep(NA,length(z.pos.effective)+1)
    betas.draws.nonOVF[,,b] = rt.output$betas
    
    #Accounting for eventual VI
    if(VI.rep>0){
      pred.kf[b,-rando.vec,]=rt.output$fitted.shu[-rando.vec,]
      betas.shu= ((b-1)/b)*betas.shu + (1/b)*rt.output$betas.shu
      rt.output$betas.shu[in.out==0,,]=rep(NA,length(z.pos.effective)+1)
      betas.shu.nonOVF= betas.shu.nonOVF+ rt.output$betas.shu
    }
    
  } #end of b-loop
  
  #proper averaging for nonOVF measures
  howmany.in = apply(whos.in.mat,1,sum)
  avg.beta.nonOVF = avg.beta.nonOVF/t(repmat(howmany.in,length(z.pos.effective)+1,1))
  for(kk in 1:dim(betas.shu.nonOVF)[3]){betas.shu.nonOVF[,,kk] = betas.shu.nonOVF[,,kk]/t(repmat(howmany.in,length(z.pos.effective)+1,1))}
  
  ###################################################################################
  ###################################################################################
  ########################## VI #####################################################
  ###################################################################################
  ###################################################################################
  if(!random.z){
    if(VI.rep>0){
      
      if(bootstrap.opt!=0){
        #get mean pred of shuffled models
        oob.pred.shu=apply(X=pred.kf[,,],FUN=mean,MARGIN=c(2,3),na.rm=TRUE)
        # mean perfo
        oob.crit.shu = rep(NA,(length(x.pos)+1)) #+length(unique(g.vec))
        normalized.target=as.matrix(scale(data[,y.pos])[1:nrow(data)])
        
        #get increase in RMSE when excluding k, out-of-bag
        for(k in 1:(length(x.pos)+1)){ #+length(unique(g.vec))
          oob.crit.shu[k] = sqrt(mean((oob.pred.shu[-oos.pos,k]-normalized.target[-oos.pos])^2,na.rm=TRUE)) #full-blown model
        }
        VI_oob=oob.crit.shu/oob.crit.shu[1]-1
        which.oob=which.min.n(-VI_oob[1:length(x.pos)], keep.VI)
        VI_oob=VI_oob[-1]
        which.all = unique(c(which.oob))
      }else{ #override since cannot do OOB VI without leaving some overvations out
        VI_oob=NULL
        which.oob=NULL
        which.all=NULL
        print('Cannot have VI-OOB without some form of Bootstrap.')
      }
      
      if(length(oos.pos)>0 & !oos.flag){
        #get increase in RMSE when excluding k, out-of-sample (as defined by oos.obs)
        poos.crit.shu = rep(NA,(length(x.pos)+1)) #+length(unique(g.vec))
        for(k in 1:(length(x.pos)+1)){ #+length(unique(g.vec))
          poos.crit.shu[k] = sqrt(mean((oob.pred.shu[oos.pos,k]-normalized.target[oos.pos])^2,na.rm=TRUE)) #full-blown model
        }
        VI_poos=poos.crit.shu/poos.crit.shu[1]-1
        VI_poos=VI_poos[-1]
        
        #keeping the most useful 'Keep.VI' members of S
        keep.VI=min(c(keep.VI,length(VI_poos)))
        which.poos=which.min.n(-VI_poos[1:length(x.pos)], keep.VI)
      }else{
        VI_poos=NULL
        which.poos=NULL
        print('Cannot have VI-OOS without an OOS.')
      }
      if(!oos.flag){which.all = unique(c(which.poos,which.oob))} #combine them
      #else{which.all = unique(c(which.oob))}
      
      ######### #VI for betas #############
      betas.shu.crit=matrix(NA,length(z.pos)+1,length(x.pos)+1) #+length(unique(g.vec))
      for(jj in 1:(length(z.pos)+1)){
        for(k in 2:(length(x.pos)+1)){ #+length(unique(g.vec))
          betas.shu.crit[jj,k] = sqrt(mean((betas.shu[,jj,k]-betas.shu[,jj,1])^2,na.rm=TRUE)) #full-blown model
        }}
      VI_betas=betas.shu.crit[,-1]
      
      betas.shu.crit=matrix(NA,length(z.pos)+1,length(x.pos)+1) #+length(unique(g.vec))
      for(jj in 1:(length(z.pos)+1)){
        for(k in 2:(length(x.pos)+1)){ #+length(unique(g.vec))
          betas.shu.crit[jj,k] = sqrt(mean((betas.shu.nonOVF[,jj,k]-betas.shu.nonOVF[,jj,1])^2,na.rm=TRUE)) #full-blown model
        }}
      VI_betas.nonOVF=betas.shu.crit[,-1]
      
      for(jj in 1:(length(z.pos)+1)){
        which.betas=which.min.n(-VI_betas.nonOVF[jj,], max(floor(keep.VI/2),1)) #used to be without nonOVF
        which.all=unique(append(which.all,which.betas))
      }
      impZ=data[,x.pos[which.all]]
    }
    else{
      VI_oob=NULL
      VI_poos=NULL
      VI_betas=NULL
      impZ=NULL
      VI_betas.nonOVF=NULL
    }
  }
  ##########################################################################################
  ##########################################################################################
  if(oos.flag){
    #cut beta time series
    avg.beta.nonOVF=avg.beta.nonOVF[-fake.pos,]
    avg.beta=avg.beta[-fake.pos,]
    betas.draws.nonOVF =betas.draws.nonOVF[-fake.pos,,] # !!!!
    betas.draws=betas.draws[-fake.pos,,] # !!!!!!
    
    #cancel VI_POOS
    VI_poos=NULL
    
    #cancel pred and the commitee
    avg.pred=NULL
    commitee=NULL
    
    #GET BACK
    data=data[-fake.pos,]
  }
  
  if(random.z==TRUE){
    VI_betas=NULL
    VI_betas.nonOVF=NULL
    VI_oob=NULL
    VI_poos=NULL
    impZ=NULL
    avg.beta=NULL
    avg.beta.nonOVF=NULL
    betas.draws = NULL
    betas.draws.nonOVF=NULL
  }
  else{ #fast graph of GTVPs
    if(cheap.look.at.GTVPs){
      if(B*(1-BS4.frac)<30){print('Warning: those bands may have missing values or be innacurate if B is low and subsampling.rate is high.')}
      bands = array(NA,dim=c(nrow(data),length(z.pos)+1,2))
      bands[,,1]=avg.beta.nonOVF
      bands[,,2]=avg.beta.nonOVF
      sd.choice=1
      
      #GET THE BANDS
      for(t in 1:nrow(data)){
        for(k in 1:(length(z.pos)+1)){
          #bands[t,k,1]=bands[t,k,1]-sd.choice*sd(betas.draws.nonOVF[t,k,],na.rm=TRUE)
          #bands[t,k,2]=bands[t,k,2]+sd.choice*sd(betas.draws.nonOVF[t,k,],na.rm=TRUE)
          bands[t,k,1]=quantile(betas.draws.nonOVF[t,k,],.16,na.rm=TRUE)
          bands[t,k,2]=quantile(betas.draws.nonOVF[t,k,],.84,na.rm=TRUE)
        }}
      
      col.vec=c("#386cb0","#fdb462","#7fc97f","#7fc97f")
      if(length(z.pos)+1>2){par(mgp=c(2.2,0.45,0), tcl=-0.4, mar=c(1.4,1.4,1.1,1.1),mfrow=c(2,2))}
      else{par(mgp=c(2.2,0.45,0), tcl=-0.4, mar=c(1.4,1.4,1.1,1.1),mfrow=c(1,2))}
      #data=as.data.frame(data)
      data=as.matrix(data)
      #print(tail(data[,c(y.pos,z.pos)]))
      #print('stucitte')
      keep.OLS= solve(crossprod(cbind(1,data[,z.pos])))%*%crossprod(cbind(1,data[,z.pos]),data[,y.pos]) # coef(lm(data[,y.pos]~data[,z.pos]))
      #print('stucitte')
      for(k in 1:(length(z.pos)+1)){
        ts.plot(cbind(avg.beta.nonOVF[,k],bands[,k,1:2]),col=col.vec[c(1,2,2)],lwd=2*c(1.1,.75,.75))
        abline(h=keep.OLS[k],col=col.vec[4],lwd=1)
        if(!oos.flag){abline(v=min(oos.pos),col=1,lwd=0.55,lty=2)}
        if(!oos.flag){legend("bottomleft", c('Posterior Mean','16 & 84% quantiles','OLS','OOS start'), col=c(col.vec[c(1,2,4)],1), lty=c(1,1,1,2), cex=.65)}
        else{legend("bottomleft", c('Posterior Mean','16 & 84% quantiles','OLS'), col=c(col.vec[c(1,2,4)]), lty=c(1,1,1), cex=.65)}
      }
      
      par(mfrow=c(1,1))
    }
  }
  
  if(!keep.forest){
    forest=NULL
    random.vecs=NULL
    }
  
  return(list(YandX=data[,c(y.pos,z.pos)],pred.ensemble=commitee,pred=avg.pred,
              betas.raw=avg.beta,
              important.S=impZ,S.names=colnames(data[,x.pos]),
              betas=avg.beta.nonOVF,
              betas.draws.raw=betas.draws,
              betas.draws=betas.draws.nonOVF,VI_betas=VI_betas.nonOVF,
              VI_oob=VI_oob,VI_oos=VI_poos,VI_betas.raw=VI_betas,
              model=list(forest=forest,data=data, regul.lambda=regul.lambda,
                         prior.var=prior.var,
                         prior.mean=prior.mean,
                         rw.regul=rw.regul,
                         HRW=HRW,
                         no.rw.trespassing=no.rw.trespassing,
                         B=B,
                         random.vecs=random.vecs,
                         y.pos=y.pos,
                         S.pos=x.pos,
                         x.pos=z.pos)
  ))
}


one.mrf.tree <- function(data,y.pos=1, x.pos,z.pos,minsize,frac=1/3,oos.pos,min.leaf.fracz=1,rw.regul,VI.rep=10,
                         ET=FALSE,ET.rate=0,fast.rw=FALSE, #g.vec=NULL,
                         #random.cuts=FALSE,Hmax=8,it.forecast=FALSE,balancing=0,
                         prior.var=c(),prior.mean=c(),prob.vec=NULL,
                         rando.vec=1:nrow(data[-oos.pos,]),trend.pos=length(x.pos),
                         skip.VI.individual=FALSE, #RV=0,ortho.trick=FALSE,lars.options=c(0,2),
                         trend.push=1,no.rw.trespassing=FALSE,
                         regul.lambda=0.01,HRW=0.25){
  
  #The original basis for this code is taken from publicly available code for a simple tree by André Bleier.
  
  #standardize data (remeber, we are doing ridge in the end)
  std.stuff = standard(data)
  data = std.stuff$Y
  
  #adjust prior.mean according to standarization
  if(!is.null(prior.mean)){
    prior.mean[-1] = (1/std.stuff$std[y.pos])*(prior.mean[-1]*std.stuff$std[z.pos])
    prior.mean[1] = (prior.mean[1]-std.stuff$mean[y.pos]+std.stuff$mean[z.pos]%*%prior.mean[-1])/std.stuff$std[y.pos]
  }
  
  #just a simple/convenient way to detect you're using BB or BBB rather than sub-sampling
  if(min(rando.vec)<1){
    weigths=rando.vec
    rando.vec = 1:(min(oos.pos)-1)
    bayes=TRUE
  }
  else{bayes=FALSE}
  
  
  if(minsize<2*min.leaf.fracz*(length(z.pos)+2)){ #to avoid you specifying a wayfor the algo to get stuck.
    minsize=2*min.leaf.fracz*(length(z.pos)+1)+2
    #  print('Minsize imposed because too small wrt z.pos length.')
  }
  
  # coerce to data.frame
  data.ori=as.data.frame(data)
  noise = 0.00000015*rnorm(nrow(data[rando.vec,])) #to avoid redundancy some bug
  data[rando.vec,]=data[rando.vec,] +noise
  
  #keep selected obs (from subsampling)
  data <- as.data.frame(data[c(rando.vec),])
  
  #data for RF regularization
  rw.regul.dat <- as.data.frame(data.ori[-oos.pos,c(1,z.pos)])
  
  # get the design matrix
  X <- cbind(1,data[,c(x.pos)])
  
  # extract target
  y <- data[, y.pos]
  
  # extract z
  z <- data[,z.pos]
  if(bayes){
    z=as.data.frame(weigths*z)
    y= as.data.frame(weigths*y)
  }
  
  # initialize while loop
  do_splits <- TRUE
  
  # create output data.frame with splitting rules and observations
  tree_info <- data.frame(NODE = 1, NOBS = nrow(data), FILTER = NA,
                          TERMINAL = "SPLIT", b0=matrix(0,1,length(z.pos)+1),   #this saves splitting information about the trees as well as full-parent nodes coefficients (before splitting), which is used by HRW (if activated)
                          stringsAsFactors = FALSE)
  
  # keep splitting until there are only leafs left
  while(do_splits) {
    
    # which parents have to be splitted
    to_calculate <- which(tree_info$TERMINAL == "SPLIT")
    #print(to_calculate)
    all.stop.flags=c()
    
    for (j in to_calculate) {
      
      # handle root node
      if (!is.na(tree_info[j, "FILTER"])) {
        # subset data according to the filter
        this_data <- subset(data, eval(parse(text = tree_info[j, "FILTER"])))
        find.out.who = subset(cbind(rando.vec,data), eval(parse(text = tree_info[j, "FILTER"])))
        whoswho=find.out.who[,1]
        
        # get the design matrix
        X <- cbind(1,this_data[,x.pos])
        y <- this_data[,y.pos]
        z <-  this_data[,z.pos]
        if(bayes){
          z=as.data.frame(weigths[whoswho]*z)
          y= as.data.frame(weigths[whoswho]*as.matrix(y))
        }
      }else{
        this_data <- data
        whoswho=rando.vec
        if(bayes){this_data <- weigths*data}
      }
      old_b0=tree_info[j, "b0"]
      
      ############## Select potential candidates for this split ###############
      SET=X[,-1]  #all X's but the intercept
      #if(y.pos<trend.pos){trend.pos=trend.pos-1} #so the user can specify trend pos in terms of position in the data matrix, not S_t
      #modulation option
      if(is.null(prob.vec)){prob.vec=rep(1,ncol(SET))}
      if(trend.push>1){prob.vec[trend.pos]=trend.push}
      
      # classic mtry move
      select.from = base::sample(x=1:ncol(SET),size=round(ncol(SET)*frac),prob=prob.vec)
      if(ncol(SET)<5){select.from=1:ncol(as.matrix(SET))}
      #########################################################################
      
      ############## Splitter function block #################################
      splitting <-  apply(SET[,select.from],  MARGIN = 2, FUN = splitter.mrf, y = y,z=z,HRW=HRW,rw.regul=rw.regul,rando.vec=rando.vec,fast.rw=fast.rw,prior.mean=prior.mean,prior.var=prior.var,
                          regul.lambda=regul.lambda,min.leaf.fracz=min.leaf.fracz,minsize=minsize,b0=old_b0,ET=ET,ET.rate=ET.rate,no.rw.trespassing=no.rw.trespassing,
                          whoswho=whoswho,regul.dat=rw.regul.dat)
      
      #########################################################################
      
      stop.flag=all(splitting[1,]==Inf)
      
      # get the min SSE
      tmp_splitter <- which.min(splitting[1,])
      
      # define maxnode
      mn <- max(tree_info$NODE)
      
      # paste filter rules
      tmp_filter <- c(paste(names(tmp_splitter), ">=",
                            splitting[2,tmp_splitter]),
                      paste(names(tmp_splitter), "<",
                            splitting[2,tmp_splitter]))
      
      # Error handling! check if the splitting rule has already been invoked
      split_here  <- !sapply(tmp_filter,
                             FUN = function(x,y) any(grepl(x, x = y)),
                             y = tree_info$FILTER)
      
      # append the splitting rules
      if (!is.na(tree_info[j, "FILTER"])) {
        tmp_filter  <- paste(tree_info[j, "FILTER"],
                             tmp_filter, sep = " & ")
      }
      
      # get the number of observations in current node
      tmp_nobs <- sapply(tmp_filter,
                         FUN = function(i, x) {
                           nrow(subset(x = x, subset = eval(parse(text = i))))
                         },
                         x = this_data)
      
      # insufficient minsize for split
      if (any(tmp_nobs <= minsize)) {
        split_here <- rep(FALSE, 2)
      }
      
      split_here <- rep(FALSE, 2)
      split_here[tmp_nobs >= minsize]=TRUE
      
      # create children data frame
      terminal = rep("SPLIT", 2)
      terminal[tmp_nobs<minsize]="LEAF"
      terminal[tmp_nobs==0]="TRASH"
      
      if(!stop.flag){
        children <- data.frame(NODE = c(mn+1, mn+2),
                               NOBS = tmp_nobs,
                               FILTER = tmp_filter,
                               TERMINAL = terminal,
                               b0=(t(cbind(as.numeric(splitting[3:(3+length(z.pos)),tmp_splitter]),
                                           as.numeric(splitting[3:(3+length(z.pos)),tmp_splitter])))),
                               row.names = NULL)
      }else{children=c()}
      
      # overwrite state of current node
      tree_info[j, "TERMINAL"] <- "PARENT"
      if(stop.flag){tree_info[j, "TERMINAL"]="LEAF"}
      
      # bind everything
      tree_info <- rbind(tree_info, children)
      
      # check if there are any open splits left
      do_splits <- !all(tree_info$TERMINAL != "SPLIT")
      
      all.stop.flags =append(all.stop.flags,stop.flag)
      
    } # end for
    if(all(all.stop.flags)){do_splits=FALSE}
  } # end while
  
  
  ###################################################################################
  ###################################################################################
  ########################## Prediction given the tree ##############################
  ###################################################################################
  ###################################################################################
  
  # extract target
  y <- data.ori[, y.pos]
  
  # extract z
  z <- cbind(data.ori[,z.pos])
  z = cbind(1,z)
  
  # calculate fitted values
  leafs <- tree_info[tree_info$TERMINAL == "LEAF", ]
  
  pga=pred.given.tree(leafs=leafs,data.ori=data.ori,oos.pos=oos.pos,z.pos=z.pos,y=y,z=z,
                      regul.lambda=regul.lambda,
                      rando.vec=rando.vec,no.rw.trespassing = no.rw.trespassing,
                      prior.var=prior.var,prior.mean=prior.mean,
                      rw.regul=rw.regul,HRW=HRW,rw.regul.dat=rw.regul.dat)
  
  beta.bank=pga$beta.bank
  fitted = pga$fitted
  
  ###################################################################################
  ###################################################################################
  ########################## Back to original units #################################
  ###################################################################################
  ###################################################################################
  
  #careful with that axe, eugene
  fitted.scaled=fitted
  fitted = fitted*std.stuff$std[y.pos] + std.stuff$mean[y.pos]
  betas= beta.bank
  betas[,1] = beta.bank[,1]*std.stuff$std[y.pos] + std.stuff$mean[y.pos]
  for(kk in 2:ncol(betas)){
    betas[,kk] = beta.bank[,kk]*std.stuff$std[y.pos]/std.stuff$std[z.pos[kk-1]]
    betas[,1]=betas[,1]-betas[,kk]*std.stuff$mean[z.pos[kk-1]] #20/09/01 correction
  }
  
  ###################################################################################
  ###################################################################################
  ########################## VI #####################################################
  ###################################################################################
  ###################################################################################
  
  if(VI.rep>0){
    #find out the X that were used -- quite useful when S_t is large, to reduce VI Compu demand
    whos.in = rep(NA,length(x.pos))
    for(k in 1:length(x.pos)){whos.in[k]=any(grepl(colnames(data.ori)[x.pos[k]],leafs$FILTER,fixed=F)==TRUE)}
    
    beta.bank.shu = array(0,dim=c(dim(beta.bank),length(x.pos)+1))
    fitted.shu = array(0,dim=c(length(fitted),length(x.pos)+1))
    
    #This calcuate what happens to betas and the prediction when a certain element of S is shuffled
    for(k in 1:length(x.pos)){
      if(whos.in[k]){
        for(ii in 1:VI.rep){
          data.shu = data.ori
          data.shu[,x.pos[k]]= sample(x=data.ori[,x.pos[k]],replace = FALSE,size=nrow(data.ori))
          
          pga=pred.given.tree(leafs=leafs,data.ori=data.shu,oos.pos=oos.pos,z.pos=z.pos,y=y,z=z,
                              regul.lambda=regul.lambda,prior.mean=prior.mean,
                              prior.var=prior.var,rw.regul=rw.regul,HRW=HRW,rw.regul.dat=rw.regul.dat)
          
          beta.bank.shu[,,k+1]=((ii-1)/ii)*beta.bank.shu[,,k+1]+pga$beta.bank/ii
          fitted.shu[,k+1] = ((ii-1)/ii)*fitted.shu[,k+1] + pga$fitted/ii
        }}
      else{
        beta.bank.shu[,,k+1]=beta.bank
        fitted.shu[,k+1]=fitted.scaled
      }
    }
    #the real fit and betas
    beta.bank.shu[,,1]=beta.bank
    fitted.shu[,1]=fitted.scaled
  }
  else{
    beta.bank.shu = array(0,dim=c(dim(beta.bank),length(x.pos)+1))
    fitted.shu = array(0,dim=c(length(fitted),length(x.pos)+1))
  }
  
  return(list(tree = tree_info[tree_info$TERMINAL == "LEAF", ], fit = fitted[-oos.pos],
              pred = fitted[oos.pos], data = data,
              betas=betas, betas.shu=beta.bank.shu,fitted.shu=fitted.shu))
}


pred.given.mrf=function(mrf.output,newdata){
  #does not work with bayes, y needs to be originally in position 1
           
  # independant of b         
  regul.lambda=mrf.output$model$regul.lambda
  prior.var=mrf.output$model$prior.var
  prior.mean=mrf.output$model$prior.mean
  rw.regul=mrf.output$model$rw.regul
  HRW=mrf.output$model$HRW
  no.rw.trespassing=mrf.output$model$no.rw.trespassing
  B=mrf.output$model$B
  data=mrf.output$model$data
  std.stuff = standard(data)
  data = std.stuff$Y
  y.pos=mrf.output$model$y.pos
  x.pos=mrf.output$model$S.pos
  z.pos=mrf.output$model$x.pos
  forest=mrf.output$model$forest
  random.vecs=mrf.output$model$random.vecs
  rownames(data)=c()
  
  if(is.null(forest[[1]])){
    stop('Need to activate keep.forest in MRF function first.')
  }
  if(any(colnames(data)=="")){
    stop('Put names on your variables with colnames, for both MRF function and this one.')
  }
  #std.stuff$std[y.pos]
  #std.stuff$mean[y.pos]
  
  data.old=data
  if(ncol(as.matrix(newdata))==1){
    oos.one.flag=TRUE
    newdata=t(as.matrix(newdata))
    newdata=rbind(newdata,newdata)
    rownames(newdata)=c()
  }else{oos.one.flag=FALSE}
  
  for(jj in 1:ncol(newdata)){
    newdata[,jj] = (newdata[,jj]- std.stuff$mean[jj+1])/std.stuff$std[jj+1]
  }
  
  
  y=rep(NA,nrow(newdata))
  data = rbind(data,cbind(y,newdata))
  colnames(data)=colnames(data.old)
  oos.pos = (nrow(data.old)+1):nrow(data)
  data.ori=as.data.frame(data)
  
  #adjust prior.mean according to standarization
  if(!is.null(prior.mean)){
    prior.mean[-1] = (1/std.stuff$std[y.pos])*(prior.mean[-1]*std.stuff$std[z.pos])
    prior.mean[1] = (prior.mean[1]-std.stuff$mean[y.pos]+std.stuff$mean[z.pos]%*%prior.mean[-1])/std.stuff$std[y.pos]
  }
  
  #data for RF regularization
  rw.regul.dat <- as.data.frame(data[-oos.pos,c(1,z.pos)])
  X <- cbind(1,data[,c(x.pos)]) #S part
  y <- data[, y.pos]
  z <- cbind(1,data[,z.pos]) #linear part
  
  avg.pred=0
  for(b in 1:B){ ##########################################################

  leafs <- forest[[b]] #tree_info[tree_info$TERMINAL == "LEAF", ]
  fitted <- rep(NA,length(y))
  beta.bank=matrix(NA,length(y),ncol(z))
  rando.vec = random.vecs[[b]]
    
  for (i in seq_len(nrow(leafs))) {
    # extract index
    #print(colnames(data.ori))
    ind.all <- as.numeric(rownames(subset(data.ori[,], eval(parse(text = leafs[i, "FILTER"])))))
    ind.all=ind.all[!is.na(ind.all)]
    ind = ind.all[!is.element(ind.all,oos.pos)] #in-sample obs of that leaf
    if(length(ind.all)>0){
      #baseline
      yy = y[ind]
      if(length(ind)==1){
        zz = t(as.matrix(z[ind,]))
        zz.all = as.matrix(z[ind.all,])
        if(length(ind.all)==1){zz.all = t(as.matrix(z[ind.all,]))}
      }else{
        zz = as.matrix(z[ind,])
        zz.all = as.matrix(z[ind.all,])
      }
      
      #simple ridge prior
      reg.mat=diag((length(z.pos)+1))*regul.lambda
      reg.mat[1,1]=0.01*reg.mat[1,1]
      
      #adds RW prior in the mix
      if(rw.regul>0){ #additional step -- creation of neighboring data
        everybody = unique(append(ind+1,ind-1)) #find.out.who+2, find.out.who-2)
        everybody = everybody[!is.element(everybody,ind.all)]
        everybody=everybody[everybody>0]
        everybody=everybody[everybody<nrow(rw.regul.dat)+1]
        if(no.rw.trespassing){everybody=intersect(everybody,rando.vec)}
        
        everybody2 = unique(append(ind+2,ind-2)) #find.out.who+2, find.out.who-2)
        everybody2 = everybody2[!is.element(everybody2,ind.all)]
        everybody2=everybody2[everybody2>0]
        everybody2=everybody2[everybody2<nrow(rw.regul.dat)+1]
        everybody2=everybody2[!is.element(everybody2,everybody)]
        if(no.rw.trespassing){everybody2=intersect(everybody2,rando.vec)}
        
        if(length(everybody)==0){
          y.neighbors=NULL
          z.neighbors=NULL
        }else{
          y.neighbors = as.matrix(rw.regul*rw.regul.dat[everybody,1])
          z.neighbors = as.matrix(rw.regul*cbind(rep(1,length(everybody)),as.matrix(rw.regul.dat[everybody,2:ncol(rw.regul.dat)])))
        }
        if(length(everybody2)==0){
          y.neighbors2=NULL
          z.neighbors2=NULL
        }else{
          y.neighbors2 = as.matrix((rw.regul^2)*rw.regul.dat[everybody2,1])
          z.neighbors2 = as.matrix((rw.regul^2)*cbind(rep(1,length(everybody2)),as.matrix(rw.regul.dat[everybody2,2:ncol(rw.regul.dat)])))
        }
        
        yy=append(append(yy,y.neighbors),y.neighbors2)
        # zz.old=zz
        # if(all(dim(zz)!=dim(z.neighbors))){print(t(as.matrix(zz)))
        #   print(z.neighbors)
        #   print(z.neighbors2)}
        if(length(zz)==(length(z.pos)+1)){
          zz=rbind(t(as.matrix(zz)),(z.neighbors),(z.neighbors2))}
        else{zz=rbind(as.matrix(zz),(z.neighbors),(z.neighbors2))}
      }
      
      #adds custom BVAR-like prior in the mix
      if(!is.null(prior.var)){
        reg.mat = diag(c(prior.var))*regul.lambda
        prior.mean.vec = prior.mean #c(0,prior.mean,rep(0,ncol(zz)-2))
        beta_hat=solve(crossprod(zz)+reg.mat,crossprod(zz,yy-zz[,]%*%prior.mean.vec))+prior.mean.vec
        b0=t(as.matrix(leafs[i,5:(5+length(z.pos))]))
      }else{
        # if(length(zz.old)==(length(z.pos)+1)){
        # print(zz)
        # print(crossprod(zz))
        # }
        #print('before')
        #print(zz)
        #print(yy)
        #print(reg.mat)
        beta_hat=solve(crossprod(zz)+reg.mat,crossprod(zz,yy))
        #print('after')
        b0=t(as.matrix(leafs[i,5:(5+length(z.pos))]))
      }
      
      #predicted values +fitted values, and, at least, the betas
      if(length(ind.all)==1){
        if(nrow(as.matrix(t(zz.all)))!=1){zz.all=t(zz.all)}
        fitted[ind.all]=t(zz.all)%*%((1-HRW)*beta_hat+HRW*b0)
      }else{
        if(ncol(as.matrix(zz.all))!=length(b0)){zz.all=t(zz.all)}
        fitted[ind.all]=zz.all%*%((1-HRW)*beta_hat+HRW*b0)
      }
      beta.bank[ind.all,]= repmat(t(((1-HRW)*beta_hat+HRW*b0)),length(ind.all),1)
    }}

  ###################################################################################
  ###################################################################################
  ########################## Back to original units #################################
  ###################################################################################
  ###################################################################################
  
  #careful with that axe, eugene
  fitted.scaled=fitted
  fitted = fitted*std.stuff$std[y.pos] + std.stuff$mean[y.pos]
  betas= beta.bank
  betas[,1] = beta.bank[,1]*std.stuff$std[y.pos] + std.stuff$mean[y.pos]
  for(kk in 2:ncol(betas)){
    betas[,kk] = beta.bank[,kk]*std.stuff$std[y.pos]/std.stuff$std[z.pos[kk-1]]
    betas[,1]=betas[,1]-betas[,kk]*std.stuff$mean[z.pos[kk-1]] #20/09/01 correction
  }
  
  #stack the trees prediction and update the mean prediction
  avg.pred = ((b-1)/b)*avg.pred +  (1/b)*fitted[oos.pos]
  
  } ###################################################################
  
  if(oos.one.flag){avg.pred=avg.pred[-2]}
  
  return(avg.pred)
}



pred.given.tree=function(leafs,data.ori,oos.pos,z.pos,y,z,regul.lambda,prior.var,prior.mean,
                         rw.regul,HRW,rw.regul.dat,rando.vec,no.rw.trespassing=FALSE){
  
  fitted <- rep(NA,length(y))
  beta.bank=matrix(NA,length(y),ncol(z))
  
  for (i in seq_len(nrow(leafs))) {
    # extract index
    ind.all <- as.numeric(rownames(subset(data.ori[,], eval(parse(text = leafs[i, "FILTER"])))))
    ind = ind.all[!is.element(ind.all,oos.pos)] #in-sample obs of that leaf
    if(length(ind.all)>0){
      #baseline
      yy = y[ind]
      if(length(ind)==1){
        zz = t(as.matrix(z[ind,]))
        zz.all = as.matrix(z[ind.all,])
        if(length(ind.all)==1){zz.all = t(as.matrix(z[ind.all,]))}
      }else{
        zz = as.matrix(z[ind,])
        zz.all = as.matrix(z[ind.all,])
      }
      
      #simple ridge prior
      reg.mat=diag((length(z.pos)+1))*regul.lambda
      reg.mat[1,1]=0.01*reg.mat[1,1]
      
      #adds RW prior in the mix
      if(rw.regul>0){ #additional step -- creation of neighboring data
        everybody = unique(append(ind+1,ind-1)) #find.out.who+2, find.out.who-2)
        everybody = everybody[!is.element(everybody,ind.all)]
        everybody=everybody[everybody>0]
        everybody=everybody[everybody<nrow(rw.regul.dat)+1]
        if(no.rw.trespassing){everybody=intersect(everybody,rando.vec)}
        
        everybody2 = unique(append(ind+2,ind-2)) #find.out.who+2, find.out.who-2)
        everybody2 = everybody2[!is.element(everybody2,ind.all)]
        everybody2=everybody2[everybody2>0]
        everybody2=everybody2[everybody2<nrow(rw.regul.dat)+1]
        everybody2=everybody2[!is.element(everybody2,everybody)]
        if(no.rw.trespassing){everybody2=intersect(everybody2,rando.vec)}
        
        if(length(everybody)==0){
          y.neighbors=NULL
          z.neighbors=NULL
        }else{
          y.neighbors = as.matrix(rw.regul*rw.regul.dat[everybody,1])
          z.neighbors = as.matrix(rw.regul*cbind(rep(1,length(everybody)),as.matrix(rw.regul.dat[everybody,2:ncol(rw.regul.dat)])))
        }
        if(length(everybody2)==0){
          y.neighbors2=NULL
          z.neighbors2=NULL
        }else{
          y.neighbors2 = as.matrix((rw.regul^2)*rw.regul.dat[everybody2,1])
          z.neighbors2 = as.matrix((rw.regul^2)*cbind(rep(1,length(everybody2)),as.matrix(rw.regul.dat[everybody2,2:ncol(rw.regul.dat)])))
        }
        
        yy=append(append(yy,y.neighbors),y.neighbors2)
        # zz.old=zz
        # if(all(dim(zz)!=dim(z.neighbors))){print(t(as.matrix(zz)))
        #   print(z.neighbors)
        #   print(z.neighbors2)}
        if(length(zz)==(length(z.pos)+1)){
          zz=rbind(t(as.matrix(zz)),(z.neighbors),(z.neighbors2))}
        else{zz=rbind(as.matrix(zz),(z.neighbors),(z.neighbors2))}
      }
      
      #adds custom BVAR-like prior in the mix
      if(!is.null(prior.var)){
        reg.mat = diag(c(prior.var))*regul.lambda
        prior.mean.vec = prior.mean #c(0,prior.mean,rep(0,ncol(zz)-2))
        beta_hat=solve(crossprod(zz)+reg.mat,crossprod(zz,yy-zz[,]%*%prior.mean.vec))+prior.mean.vec
        b0=t(as.matrix(leafs[i,5:(5+length(z.pos))]))
      }else{
        # if(length(zz.old)==(length(z.pos)+1)){
        # print(zz)
        # print(crossprod(zz))
        # }
        beta_hat=solve(crossprod(zz)+reg.mat,crossprod(zz,yy))
        b0=t(as.matrix(leafs[i,5:(5+length(z.pos))]))
      }
      
      #predicted values +fitted values, and, at least, the betas
      if(length(ind.all)==1){
        if(nrow(as.matrix(t(zz.all)))!=1){zz.all=t(zz.all)}
        fitted[ind.all]=t(zz.all)%*%((1-HRW)*beta_hat+HRW*b0)
      }else{
        if(ncol(as.matrix(zz.all))!=length(b0)){zz.all=t(zz.all)}
        fitted[ind.all]=zz.all%*%((1-HRW)*beta_hat+HRW*b0)
      }
      beta.bank[ind.all,]= repmat(t(((1-HRW)*beta_hat+HRW*b0)),length(ind.all),1)
    }}
  return(list(fitted=fitted,beta.bank=beta.bank))
}


splitter.mrf <- function(x, y,z,regul.lambda,min.leaf.fracz,whoswho,regul.dat,rw.regul,prior.var=NULL,prior.mean=NULL,
                         ET=FALSE,ET.rate=0,no.rw.trespassing=FALSE,rando.vec=c(),fast.rw=TRUE,
                         minsize,HRW,b0,cons_w=0.01,random.cuts=FALSE){
  
  uni_x = unique(x)
  splits <- sort(uni_x)
  z=cbind(1,as.matrix(z))
  y=as.matrix(y)
  sse <- rep(Inf,length(uni_x))
  the.seq = seq_along(splits)
  if(!(rw.regul>0)){fast.rw=TRUE} #impose it if rw.gul not >0
  
  #sub-selection of potential splitting point, either to randomize further or to speed things up
  if(!is.null(ET.rate)){
    if(ET==TRUE& nrow(z)>2*minsize){ #random selection of potetial splitting points for a given S
      samp=splits[min.leaf.fracz*(ncol(z)):(length(splits)-min.leaf.fracz*(ncol(z)))]
      splits=sample(x=samp,size=max(1,ET.rate*length(samp))) #pure ET is ET rate =0 so it is 1
      the.seq = seq_along(splits)
    }
    if(ET==FALSE & nrow(z)>4*minsize){ #not considering every point, considering a certain fraction (quantiles). Need for speed.
      samp=splits[(min.leaf.fracz*ncol(z)):(length(splits)-(min.leaf.fracz*ncol(z)))]
      splits = quantile(samp,probs=seq(0.01,.99, length.out=floor(max(1,ET.rate*length(samp)))))
      the.seq = seq_along(splits)
    }}
  
  #regularization matrix
  reg.mat = diag(ncol(z))*regul.lambda
  reg.mat[1,1]=cons_w*reg.mat[1,1]
  
  if(!is.null(prior.var)){
    reg.mat = diag(c(prior.var))*regul.lambda
  }
  
  ###### full parent-node betas (for hierarichal prior) #################
  #bvar or not
  if(is.null(prior.mean)){
    b0 = solve(crossprod(z)+reg.mat,crossprod(z,y)) #for the hierarchical prior
  }else{
    b0 = solve(crossprod(z)+reg.mat,crossprod(z,y-z%*%prior.mean))+prior.mean #for the hierarchical prior
  }
  #######################################################################
  nrrd=nrow(regul.dat)
  ncrd=ncol(regul.dat)
  
  for (i in the.seq) {
    sp <- splits[i]
    id1 = which(x < sp)
    id2= which(x >= sp)
    
    #print(length(y[id2])>=(min.leaf.fracz*(ncol(z))))
    if( (length(id1)>=(min.leaf.fracz*(ncol(z)))) &
        (length(id2)>=(min.leaf.fracz*(ncol(z))))){
      
      id=id1
      yy = y[id]
      zz = z[id,]
      zz.privy = zz
      #if(length(id)==1){zz = t(as.matrix(z[id,]))}
      
      if(!fast.rw){
        everybody = union(whoswho[id]+1,whoswho[id]-1) #find.out.who+2, find.out.who-2)
        everybody = setdiff(everybody,whoswho)
        everybody=everybody[everybody>0]
        everybody=everybody[everybody<nrrd+1]
        if(no.rw.trespassing){everybody=intersect(everybody,rando.vec)}
        everybody2 = union(whoswho[id]+2,whoswho[id]-2) #find.out.who+2, find.out.who-2)
        everybody2=setdiff(everybody2,whoswho)
        everybody2=setdiff(everybody2,everybody)
        everybody2=everybody2[everybody2>0]
        everybody2=everybody2[everybody2<nrrd+1]
        if(no.rw.trespassing){everybody2=intersect(everybody2,rando.vec)}
        
        if(length(everybody)==0){
          y.neighbors=NULL
          z.neighbors=NULL
        }else{
          y.neighbors = as.matrix(rw.regul*regul.dat[everybody,1])
          z.neighbors = as.matrix(rw.regul*cbind(rep(1,length(everybody)),as.matrix(regul.dat[everybody,2:ncrd])))
        }
        if(length(everybody2)==0){
          y.neighbors2=NULL
          z.neighbors2=NULL
        }else{
          y.neighbors2 = as.matrix((rw.regul^2)*regul.dat[everybody2,1])
          z.neighbors2 = as.matrix((rw.regul^2)*cbind(rep(1,length(everybody2)),as.matrix(regul.dat[everybody2,2:ncrd])))
        }
        
        yy=append(append(yy,y.neighbors),y.neighbors2)
        zz=rbind(as.matrix(zz),z.neighbors,z.neighbors2)
      }
      
      #bvar or not
      if(is.null(prior.mean)){
        if(length(yy)!=dim(zz)[1]){print(paste(length(yy),'and',dim(zz)[1]))}
        #print()
        p1=(zz.privy%*%((1-HRW)*solve(crossprod(zz)+reg.mat,crossprod(zz,yy))+HRW*b0)) #[1:length(id)]
      }else{
        p1=(zz.privy%*%((1-HRW)*solve(crossprod(zz)+reg.mat,crossprod(zz,yy-zz%*%prior.mean))+prior.mean+HRW*b0)) #[1:length(id)]
      }
      
      #part 2
      id = id2
      yy = y[id]
      zz = z[id,]
      zz.privy = zz
      #if(length(id)==1){zz = t(as.matrix(z[id,]))}
      
      if(!fast.rw){
        everybody = union(whoswho[id]+1,whoswho[id]-1) #find.out.who+2, find.out.who-2)
        everybody = setdiff(everybody,whoswho)
        everybody=everybody[everybody>0]
        everybody=everybody[everybody<nrrd+1]
        if(no.rw.trespassing){everybody=intersect(everybody,rando.vec)}
        everybody2 = union(whoswho[id]+2,whoswho[id]-2) #find.out.who+2, find.out.who-2)
        everybody2=setdiff(everybody2,whoswho)
        everybody2=setdiff(everybody2,everybody)
        everybody2=everybody2[everybody2>0]
        everybody2=everybody2[everybody2<nrrd+1]
        if(no.rw.trespassing){everybody2=intersect(everybody2,rando.vec)}
        
        if(length(everybody)==0){
          y.neighbors=NULL
          z.neighbors=NULL
        }else{
          y.neighbors = as.matrix(rw.regul*regul.dat[everybody,1])
          z.neighbors = as.matrix(rw.regul*cbind(rep(1,length(everybody)),as.matrix(regul.dat[everybody,2:ncrd])))
        }
        if(length(everybody2)==0){
          y.neighbors2=NULL
          z.neighbors2=NULL
        }else{
          y.neighbors2 = as.matrix((rw.regul^2)*regul.dat[everybody2,1])
          z.neighbors2 = as.matrix((rw.regul^2)*cbind(rep(1,length(everybody2)),as.matrix(regul.dat[everybody2,2:ncrd])))
        }
        
        yy=append(append(yy,y.neighbors),y.neighbors2)
        zz=rbind(as.matrix(zz),z.neighbors,z.neighbors2,NULL)
      }
      
      #bvar or not
      if(is.null(prior.mean)){
        p2=(zz.privy%*%((1-HRW)*solve(crossprod(zz)+reg.mat,crossprod(zz,yy))+HRW*b0)) #[1:length(id)]
      }else{
        p2=(zz.privy%*%((1-HRW)*solve(crossprod(zz)+reg.mat,crossprod(zz,yy-zz%*%prior.mean))+prior.mean+HRW*b0)) #[1:length(id)]
      }
      
      sse[i] <- sum((y[id1]-p1)^2) + sum((y[id2]-p2)^2)
    }
  }
  
  sse = DV.fun(sse,DV.pref=.15) #implement a mild preference for 'center' splits, allows trees to run deeper
  split_at <- splits[which.min(sse)]
  return(c(sse = min(sse), split = split_at,b0=b0))
}

DV.fun = function(sse,DV.pref=.25){ #implement a middle of the range preference for middle of the range splits.
  seq = seq_along(sse)
  down.voting = 0.5*seq^2 - seq
  down.voting = down.voting/mean(down.voting)
  down.voting = down.voting-min(down.voting)+1
  down.voting = down.voting^DV.pref
  return(sse*down.voting)
}

which.min.n=function(x, n = 1)
{
  if (n == 1)
    which.min(x)
  else {
    if (n > 1) {
      ii <- order(x, decreasing = FALSE)[1:min(n, length(x))]
      ii[!is.na(x[ii])]
    }
    else {
      stop("n must be >=1")
    }
  }
}

standard <- function(Y){ #from Stéphane Surprenant
  Y < as.matrix(Y)
  size <- dim(Y)
  
  mean_y <-apply(Y, c(2), mean, na.rm=TRUE)
  sd_y <- apply(Y, c(2), sd, na.rm=TRUE)
  
  Y0 <- (Y - repmat(mean_y, size[1],1))/repmat(sd_y, size[1],1)
  return(list(Y=Y0, mean=mean_y, std=sd_y))
}
