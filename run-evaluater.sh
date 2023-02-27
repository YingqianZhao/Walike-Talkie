for dType in equal nofake normal uniform
do
	for run in {0..99}
	do
		python2 evaluater.py evaluater_${dType}_${run} testedefensesf/etester_${dType}_${run}
	done
done
