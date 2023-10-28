module IsingMCTest

using ShiftedArrays
using Infiltrator
using Random
using Plots

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
	EJ = -4;
	for iJ = 1:5, iH = 1:2
		EH = (-1)^iH;
		EIsing = EJ * Jsgnd + EH * Hsgnd;
	end
	
	while true
		# x = rand(rangeCoord);
		# y = rand(rangeCoord);
		pos = rand(indLst);
		
		# @time begin
		dE = -2 * boolToIntPN(spinArr[pos]) * Hsgnd;
		rndMC = rand();
		thresMC = 1;
		for iD = 1 : nDim, iSh = 1 : 2
			lnkBool = xor( spinArr[pos], spinArrSh[iD,iSh][pos] );
			lnk = boolToIntPN( lnkBool );
			dE -= 2*Jsgnd * lnk;
		end
		# lnk = xor( spinArr[x,y], spinArr[x+1,y] );
		# dEJ = -2 * boolToIntPN();
		isFlip = false;
		if dE < 0
			spinArr[pos] = !spinArr[pos];
			isFlip = true;
		else 
			thresMC = exp( -dE );
			if rndMC < thresMC
				spinArr[pos] = !spinArr[pos];
				isFlip = true;
			end
		end
		
		
		# display(pltSpins);
		# @infiltrate
		
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
	@infiltrate
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
	@infiltrate
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
