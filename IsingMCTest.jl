module IsingMCTest

using ShiftedArrays
# using Infiltrator
using Random
using Plots

struct SpinArray
	sz::Int64;
	nDim::Int64;
	arr::Array{Bool};
	arrSh::Array{CircShiftedArray{Bool}};
	indLst::CartesianIndices;
	
	function SpinArray( sz::Int64, nDim::Int64 )
		arr = rand( Bool, ntuple( x->sz, nDim ) );
		arrSh = [ ShiftedArrays.circshift( arr, ntuple( dim -> dim == iD ? (-1)^iSh : 0, nDim ) ) for iD = 1 : nDim, iSh = 1:2 ];
		indLst = CartesianIndices(arr);
		
		new( sz, nDim, arr, arrSh, indLst );
	end
end

abstract type MCUpdater end

function mcUpdaterGenFun( updaterType::DataType, Jsgnd, Hsgnd, nDim ) 
	return updaterType( Jsgnd, Hsgnd, nDim );
end

function mcUpdateFun!( spinArrObj::SpinArray, updater::MCUpdater ) end

function isingMCMethods( sz::Int64; updaterType::DataType = MetropMCUpdater, itStop = nothing, J = 1, H = 0, itSkip = 10::Int64 )
	Jsgnd = -J;
	Hsgnd = -H;
	nDim = 2;
	spinArrObj = SpinArray( sz, nDim );
	
	mcUpdater = mcUpdaterGenFun( updaterType, Jsgnd, Hsgnd, nDim );
	
	pltSpins = heatmap( spinArrObj.arr, color = cgrad( :greys, rev=true ), legend = :none );
	display(pltSpins);
	
	it = 1;
	while true
		mcUpdateFun!( spinArrObj, mcUpdater );
		
		if !isnothing(itStop)
			if it >= itStop
				break;
			end
		end
		# print( it, ",", "          \r" )
		# end
		
		if it % itSkip == 0
			plt = heatmap( spinArrObj.arr, color = cgrad( :greys, rev=true ), legend = :none );
			display(plt);
			# sleep(0.0001);
		end
		it += 1;
	end
end

isingMCMetrop( sz::Int64; itStop = nothing, J = 1, H = 0, itSkip::Int64 = 10 ) = isingMCMethods( sz; updaterType = MetropMCUpdater, itStop = itStop, J = J, H = H, itSkip = itSkip );

struct MetropMCUpdater <: MCUpdater
	expDELst::Matrix{Float64};
	
	function MetropMCUpdater( Jsgnd, Hsgnd, nDim )
		dELst = zeros( 2*nDim+1, 2 );
		for iJ = 1:2*nDim+1, iH = 1:2
			EJ = 2 * ( iJ - *( nDim+1 ) );
			EH = (-1)^iH;
			dELst[iJ,iH] = -2 * ( EJ * Jsgnd + EH * Hsgnd );
		end
		expDELst = exp.(-dELst);
		
		new( expDELst );
	end
end

function mcUpdateFun!( spinArrObj::SpinArray, mcUpdater::MetropMCUpdater )
	pos = rand(spinArrObj.indLst);
	
	iEH = spinArrObj.arr[pos] ? 2 : 1;
	dEJ = 0;
	for iDim = 1 : spinArrObj.nDim, iSh = 1 : 2
		lnkBool = !xor( spinArrObj.arr[pos], spinArrObj.arrSh[iDim,iSh][pos] );
		dEJ += lnkBool;
	end
	iEJ = dEJ + 1;
	
	expDE = mcUpdater.expDELst[iEJ,iEH];
	if expDE >= 1
		spinArrObj.arr[pos] = !spinArrObj.arr[pos];
	else
		rndThres = rand();
		if rndThres < expDE
			spinArrObj.arr[pos] = !spinArrObj.arr[pos]
		end
	end
end

function isingMC( sz::Int64; itStop = nothing, J = 1, H = 0, itSkip = 10::Int64 )
	Jsgnd = -J;
	Hsgnd = -H;
	spinArr = rand( Bool, sz, sz );
	pltSpins = heatmap( spinArr, color = cgrad( :greys, rev=true ), legend = :none );
	display(pltSpins);
	nDim = 2;
	indLst = CartesianIndices( spinArr );
	spinArrSh = [ ShiftedArrays.circshift( spinArr, ntuple( dim -> dim == iD ? (-1)^iSh : 0 , nDim ) ) for iD = 1 : nDim, iSh = 1:2 ];
	
	it = 1;
	rangeCoord = 1:sz;
	
	dELst = zeros( 5, 2 );
	# EJ = -4;
	for iJ = 1:5, iH = 1:2
		EJ = 2*iJ - 6;
		EH = (-1)^iH;
		dELst[iJ,iH] = -2 * ( EJ * Jsgnd + EH * Hsgnd );
	end
	expDELst = exp.(-dELst);
	# @infiltrate
	
	while true
		# x = rand(rangeCoord);
		# y = rand(rangeCoord);
		pos = rand(indLst);
		
		# @time begin
		# dE = -2 * boolToIntPN(spinArr[pos]) * Hsgnd;
		idEH = spinArr[pos] ? 2 : 1;
		rndMC = rand();
		thresMC = 1;
		dEJ = 0;
		for iD = 1 : nDim, iSh = 1 : 2
			lnkBool = !xor( spinArr[pos], spinArrSh[iD,iSh][pos] );
			# lnk = boolToIntPN( lnkBool );
			dEJ += lnkBool;
		end
		idEJ = dEJ + 1;
		dE = dELst[idEJ, idEH];
		# @infiltrate
		
		# lnk = xor( spinArr[x,y], spinArr[x+1,y] );
		# dEJ = -2 * boolToIntPN();
		isFlip = false;
		if dE <= 0
			spinArr[pos] = !spinArr[pos];
			isFlip = true;
		else 
			# thresMC = exp( -dE );
			thresMC = expDELst[idEJ,idEH];
			if rndMC < thresMC
				spinArr[pos] = !spinArr[pos];
				isFlip = true;
			end
		end
		
		if !isnothing(itStop)
			if it >= itStop
				break;
			end
		end
		# print( it, ",", isFlip, ", ", pos, ", ", dE, ",", "          \r" )
		# end
		
		if it % itSkip == 0
			plt = heatmap( spinArr, color = cgrad( :greys, rev=true ), legend = :none );
			display(plt);
			# sleep(0.0001);
			# @infiltrate
		end
		it += 1;
	end
	# @infiltrate
end

function pltUpdateSine( itStop = 50 )
	pyplot();
	xBase = range(0,2*pi,100);
	x = collect( xBase );
	xStep = 2*pi/100;
	y = sin.(x);
	p1 = plot(x,y);
	display(p1);
	for it = 1 : itStop
		x .+= xStep;
		y .= sin.(x);
		p2 = plot(x,y);
		display(p2);
		sleep(0.0001);
		# @infiltrate
	end
	# @infiltrate
end

function boolToIntPN( valBool::Bool )
	return valBool ? -1 : 1;
end

function testSh()
	alst = rand(Bool, 3,3);
	# @infiltrate
	pyplot();
	alstSh = ShiftedArrays.circshift( alst, (-1,0) );
	p1 = heatmap( alst, color = cgrad( :greys, rev=true ), legend = :none )
	p2 = heatmap( alstSh, color = cgrad( :greys, rev=true ), legend = :none, reuse = false )
	display(p1);
	display(p2);
	readline();
end

function boolAndToIntPN( val1::Bool, val2::Bool )
	return boolToIntPN( xor(val1, val2) );
end
	
end
