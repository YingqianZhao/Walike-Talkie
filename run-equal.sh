for dType in nofake equal normal uniform
do
	for run in {0..25}
	do
		python2 equal_tester.py etester_${dType}_${run} $dType
	done
done
