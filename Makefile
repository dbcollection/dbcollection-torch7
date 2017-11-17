install:
	luarocks install rocks/*

uninstall:
	luarocks remove dbcollection

test:
	th tests/test.lua