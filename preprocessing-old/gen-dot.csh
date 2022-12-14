#! /bin/csh -f

foreach file (*.dot)
	dot -Tpng "$file" -o "$file.png"
end
