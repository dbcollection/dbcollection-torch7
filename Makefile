install:
	luarocks make rocks/*

uninstall:
	luarocks remove dbcollection

test:
	make install
	th tests/test.lua
